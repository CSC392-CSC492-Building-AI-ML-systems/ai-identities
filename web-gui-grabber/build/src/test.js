import { chromium } from 'patchright';
import path from 'path';
import fs from 'fs';
import { Worker } from 'worker_threads';
const args = process.argv.slice(2); // skip the first two default entries
const website = args[0];
const num_workers = parseInt(args[1]);
const base_profile = args[2];
const profiles_clones_dir = args[3];
const eventLogs = [];
// 10 prompts total - 2 per worker
const promptArrTxt = fs.readFileSync(args[4], 'utf-8');
const ALL_PROMPTS = JSON.parse(promptArrTxt);
console.log(args);
const waitDuration = parseInt(args[5]);
console.log(waitDuration);
let hasRegistered = false;
async function createProfileCopies() {
    if (!fs.existsSync(profiles_clones_dir)) {
        fs.mkdirSync(profiles_clones_dir, {
            recursive: true
        });
    }
    const dirs = [];
    for(let i = 0; i < num_workers; i++){
        const cloneDir = path.join(profiles_clones_dir, `profile-${i}`);
        dirs.push(cloneDir);
    }
    await Promise.all(dirs.map((cloneDir)=>fs.rm(cloneDir, ()=>{})));
    await Promise.all(dirs.map((cloneDir)=>fs.cp(base_profile, cloneDir, ()=>{})));
    return dirs;
}
function spawnWorkers(profileDirs) {
    let promptList = [];
    for (const _ of profileDirs){
        promptList.push([]);
    }
    for(let i = 0; i < ALL_PROMPTS.length; i++){
        promptList[i % profileDirs.length].push(ALL_PROMPTS[i]);
    }
    const promises = profileDirs.map((profileDir, i)=>{
        return new Promise((resolve, reject)=>{
            let workerPrompts = promptList[i];
            const worker = new Worker('./build/src/worker.js', {
                workerData: {
                    eventLogs,
                    userDataDir: profileDir,
                    workerId: i,
                    prompts: promptList[i],
                    website: website,
                    waitDuration: waitDuration
                }
            });
            console.log(`Worker ${i} started with profile ${profileDir} and prompts:`, workerPrompts);
            worker.on('message', (msg)=>{
                console.log(`Worker ${i} finished with results for ${Object.keys(msg).length} prompts`);
                resolve(msg);
            });
            worker.on('error', reject);
            worker.on('exit', (code)=>{
                if (code !== 0) reject(new Error(`Worker ${i} exited with code ${code}`));
            });
        });
    });
    Promise.all(promises).then((results)=>{
        // Combine all worker results into a single JSON object
        // Structure: { "prompt text": ["response1", "response2", "response3", "response4"] }
        const combinedResults = {};
        results.forEach((workerResult)=>{
            Object.keys(workerResult).forEach((prompt)=>{
                combinedResults[prompt] = workerResult[prompt];
            });
        });
        // Write results to JSON file
        const outputPath = './prompt_responses.json';
        fs.writeFileSync(outputPath, JSON.stringify(combinedResults, null, 2), 'utf8');
        console.log(`\nResults written to ${outputPath}`);
        console.log('\n=== SUMMARY ===');
        console.log(`Total workers: ${results.length}`);
        console.log(`Total prompts: ${Object.keys(combinedResults).length}`);
        console.log(`Total responses: ${Object.values(combinedResults).flat().length}`);
        console.log('===============');
        // Log a sample of results for verification
        Object.keys(combinedResults).forEach((prompt)=>{
            console.log(`\nPrompt: "${prompt}"`);
            console.log(`Responses collected: ${combinedResults[prompt].length}`);
            combinedResults[prompt].forEach((response, index)=>{
                console.log(`  ${index + 1}: ${response.substring(0, 100)}${response.length > 100 ? '' : ''}`);
            });
        });
    }).catch((error)=>{
        console.error('Error in workers:', error);
    });
}
(async ()=>{
    const context = await chromium.launchPersistentContext(base_profile, {
        channel: 'chrome',
        headless: false,
        viewport: {
            height: 0,
            width: 0
        }
    });
    async function setupEventListeners(page) {
        if (!hasRegistered) {
            await page.exposeFunction('logEvent', async (type, details)=>{
                if (!details) {
                    details = {
                        message: type
                    };
                }
                let logEntry = {
                    type,
                    details,
                    timestamp: Date.now()
                };
                eventLogs.push(logEntry);
            });
            await page.exposeFunction('closeBrowser', ()=>{
                context.close().then(async ()=>{
                    const profileDirs = createProfileCopies();
                    let worker_dirs = [];
                    for(let i = 0; i < num_workers; i++){
                        worker_dirs.push('./cloned-profiles/profile-' + i.toString());
                    }
                    spawnWorkers(await profileDirs);
                });
            });
            hasRegistered = true;
        }
        await page.evaluate(()=>{
            console.log("Attaching listeners...");
            // Remove old listeners if they exist
            if (window._mouseMoveListener) {
                document.removeEventListener('mousemove', window._mouseMoveListener);
                document.removeEventListener('click', window._clickListener);
                document.removeEventListener('dblclick', window._dblClickListener);
                document.removeEventListener('keydown', window._keyDownListener);
            }
            // Define listeners
            const mouseMoveListener = (e)=>{
                try {
                    window.logEvent('mousemove', {
                        x: e.clientX,
                        y: e.clientY
                    });
                } catch (e) {
                    console.error(e);
                    console.log("FAILED");
                    window.location.reload();
                }
            };
            const clickListener = (e)=>{
                try {
                    window.logEvent('click', {
                        x: e.clientX,
                        y: e.clientY,
                        button: e.button
                    });
                } catch (e) {
                    console.error(e);
                    console.log("FAILED");
                    window.location.reload();
                }
            };
            const dblClickListener = (e)=>{
                try {
                    window.logEvent('dblclick', {
                        x: e.clientX,
                        y: e.clientY,
                        button: e.button
                    });
                } catch (e) {
                    console.error(e);
                    console.log("FAILED");
                    window.location.reload();
                }
            };
            const copyListener = async (e)=>{
                window.logEvent('copy');
                console.log(await navigator.clipboard.readText());
            };
            const keyDownListener = (e)=>{
                if (e.key === 'F12') {
                    e.preventDefault();
                    console.log('F12 pressed - entering replay mode');
                    window.replayMode = true;
                    window.logEvent('f12_pressed', {
                        message: 'Entered replay mode - press any key to start replay'
                    });
                } else if (e.key === 'Escape') {
                    window.logEvent('escape_pressed', {
                        message: 'Closing browser'
                    });
                } else if (e.key === 'Enter') {
                    window.logEvent('enter_key_press', {
                        key: e.key
                    });
                } else if (window.replayMode) {
                    e.preventDefault();
                    window.logEvent('replay_trigger', {
                        key: e.key,
                        message: 'Starting replay...'
                    });
                    try {
                        window.closeBrowser();
                    } catch (err) {
                        console.log(err);
                    }
                    window.replayMode = false;
                }
            };
            // Save references globally to allow future removal
            window._mouseMoveListener = mouseMoveListener;
            window._clickListener = clickListener;
            window._dblClickListener = dblClickListener;
            window._keyDownListener = keyDownListener;
            // Attach new listeners
            document.addEventListener('mousemove', mouseMoveListener);
            document.addEventListener('click', clickListener);
            document.addEventListener('dblclick', dblClickListener);
            document.addEventListener('keydown', keyDownListener);
            document.addEventListener('copy', copyListener);
        });
    }
    const page = await context.newPage();
    page.on('framenavigated', async ()=>{
        try {
            await setupEventListeners(page);
        } catch (_) {
            ;
        }
    });
    await page.goto(website);
    await page.waitForLoadState('load');
    let doneReload = false;
    console.log('Loading script...');
    while(!doneReload){
        try {
            await page.reload();
            doneReload = true;
        } catch (_) {
            ;
        }
    }
    await page.setViewportSize({
        width: 1280,
        height: 720
    });
    console.log('Browser opened. Interact with the page:');
    console.log('- Record your interactions normally');
    console.log('- Press F12 to enter replay mode');
    console.log('- Press any key after F12 to start workers');
    console.log(`- Each worker will run 2 prompts, 4 times each`);
    console.log(`- Total: ${num_workers} workers × 2 prompts × 4 iterations = ${num_workers * 2 * 4} responses`);
})();

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIi4uLy4uL3NyYy90ZXN0LnRzIl0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IGNocm9taXVtLCBQYWdlIH0gZnJvbSAncGF0Y2hyaWdodCc7XG5pbXBvcnQgcGF0aCBmcm9tICdwYXRoJztcbmltcG9ydCBmcyBmcm9tICdmcyc7XG5pbXBvcnQgeyBXb3JrZXIgfSBmcm9tICd3b3JrZXJfdGhyZWFkcyc7XG5cbmRlY2xhcmUgZ2xvYmFsIHtcbiAgaW50ZXJmYWNlIFdpbmRvdyB7XG4gICAgcmVwbGF5TW9kZTogYm9vbGVhbjtcbiAgICBsb2dFdmVudDogKHR5cGU6IEV2ZW50VHlwZSwgZGV0YWlscz86IEV2ZW50RGV0YWlscykgPT4gdm9pZDtcbiAgICBjbG9zZUJyb3dzZXI6ICgpID0+IHZvaWQ7XG4gICAgXG4gIH1cbn1cblxuY29uc3QgYXJncyA9IHByb2Nlc3MuYXJndi5zbGljZSgyKTsgLy8gc2tpcCB0aGUgZmlyc3QgdHdvIGRlZmF1bHQgZW50cmllc1xuXG5jb25zdCB3ZWJzaXRlID0gYXJnc1swXVxuY29uc3QgbnVtX3dvcmtlcnMgPSBwYXJzZUludChhcmdzWzFdKTtcbmNvbnN0IGJhc2VfcHJvZmlsZSA9IGFyZ3NbMl1cbmNvbnN0IHByb2ZpbGVzX2Nsb25lc19kaXIgPSBhcmdzWzNdXG5jb25zdCBldmVudExvZ3M6IEN1c3RvbUV2ZW50TG9nW10gPSBbXTtcblxuXG5cbi8vIDEwIHByb21wdHMgdG90YWwgLSAyIHBlciB3b3JrZXJcblxuY29uc3QgcHJvbXB0QXJyVHh0ID0gZnMucmVhZEZpbGVTeW5jKGFyZ3NbNF0sICd1dGYtOCcpO1xuY29uc3QgQUxMX1BST01QVFM6IHN0cmluZ1tdID0gSlNPTi5wYXJzZShwcm9tcHRBcnJUeHQpXG5cbmNvbnNvbGUubG9nKGFyZ3MpXG5jb25zdCB3YWl0RHVyYXRpb246IG51bWJlciA9IHBhcnNlSW50KGFyZ3NbNV0pXG5jb25zb2xlLmxvZyh3YWl0RHVyYXRpb24pXG5cbmxldCBoYXNSZWdpc3RlcmVkID0gZmFsc2U7XG5cbmFzeW5jIGZ1bmN0aW9uIGNyZWF0ZVByb2ZpbGVDb3BpZXMoKTogUHJvbWlzZTxzdHJpbmdbXT4ge1xuICBcbiAgaWYgKCFmcy5leGlzdHNTeW5jKHByb2ZpbGVzX2Nsb25lc19kaXIpKSB7XG4gICAgZnMubWtkaXJTeW5jKHByb2ZpbGVzX2Nsb25lc19kaXIsIHsgcmVjdXJzaXZlOiB0cnVlIH0pO1xuICB9XG5cbiAgY29uc3QgZGlyczogc3RyaW5nW10gPSBbXTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IG51bV93b3JrZXJzOyBpKyspIHtcbiAgICBjb25zdCBjbG9uZURpciA9IHBhdGguam9pbihwcm9maWxlc19jbG9uZXNfZGlyLCBgcHJvZmlsZS0ke2l9YCk7XG5cbiAgICBkaXJzLnB1c2goY2xvbmVEaXIpO1xuICB9XG5cbiAgYXdhaXQgUHJvbWlzZS5hbGwoXG4gICAgZGlycy5tYXAoKGNsb25lRGlyKSA9PlxuICAgICAgZnMucm0oY2xvbmVEaXIsICgpPT57fSlcbiAgICApXG4gICk7XG5cbiAgYXdhaXQgUHJvbWlzZS5hbGwoXG4gICAgZGlycy5tYXAoKGNsb25lRGlyKSA9PlxuICAgICAgZnMuY3AoYmFzZV9wcm9maWxlLCBjbG9uZURpciwgKCk9Pnt9KVxuICAgIClcbiAgKTtcblxuICByZXR1cm4gZGlycztcbn1cblxuZnVuY3Rpb24gc3Bhd25Xb3JrZXJzKHByb2ZpbGVEaXJzOiBzdHJpbmdbXSkge1xuICBsZXQgcHJvbXB0TGlzdDogc3RyaW5nW11bXSA9IFtdXG5cbiAgZm9yIChjb25zdCBfIG9mIHByb2ZpbGVEaXJzKSB7XG4gICAgcHJvbXB0TGlzdC5wdXNoKFtdKVxuICB9XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBBTExfUFJPTVBUUy5sZW5ndGg7IGkrKykge1xuICAgIHByb21wdExpc3RbaSAlIHByb2ZpbGVEaXJzLmxlbmd0aF0ucHVzaChBTExfUFJPTVBUU1tpXSlcbiAgfVxuXG4gIGNvbnN0IHByb21pc2VzID0gcHJvZmlsZURpcnMubWFwKChwcm9maWxlRGlyLCBpKSA9PiB7XG4gICAgcmV0dXJuIG5ldyBQcm9taXNlPFJlY29yZDxzdHJpbmcsIHN0cmluZ1tdPj4oKHJlc29sdmUsIHJlamVjdCkgPT4ge1xuICAgICAgbGV0IHdvcmtlclByb21wdHMgPSBwcm9tcHRMaXN0W2ldXG4gICAgICBjb25zdCB3b3JrZXIgPSBuZXcgV29ya2VyKCcuL2J1aWxkL3NyYy93b3JrZXIuanMnLCB7XG4gICAgICAgIHdvcmtlckRhdGE6IHtcbiAgICAgICAgICBldmVudExvZ3MsXG4gICAgICAgICAgdXNlckRhdGFEaXI6IHByb2ZpbGVEaXIsXG4gICAgICAgICAgd29ya2VySWQ6IGksXG4gICAgICAgICAgcHJvbXB0czogcHJvbXB0TGlzdFtpXSxcbiAgICAgICAgICB3ZWJzaXRlOiB3ZWJzaXRlLFxuICAgICAgICAgIHdhaXREdXJhdGlvbjogd2FpdER1cmF0aW9uIFxuICAgICAgICB9LFxuICAgICAgfSk7XG5cbiAgICAgIGNvbnNvbGUubG9nKGBXb3JrZXIgJHtpfSBzdGFydGVkIHdpdGggcHJvZmlsZSAke3Byb2ZpbGVEaXJ9IGFuZCBwcm9tcHRzOmAsIHdvcmtlclByb21wdHMpO1xuXG4gICAgICB3b3JrZXIub24oJ21lc3NhZ2UnLCAobXNnKSA9PiB7XG4gICAgICAgIGNvbnNvbGUubG9nKGBXb3JrZXIgJHtpfSBmaW5pc2hlZCB3aXRoIHJlc3VsdHMgZm9yICR7T2JqZWN0LmtleXMobXNnKS5sZW5ndGh9IHByb21wdHNgKTtcbiAgICAgICAgcmVzb2x2ZShtc2cpO1xuICAgICAgfSk7XG5cbiAgICAgIHdvcmtlci5vbignZXJyb3InLCByZWplY3QpO1xuICAgICAgd29ya2VyLm9uKCdleGl0JywgKGNvZGUpID0+IHtcbiAgICAgICAgaWYgKGNvZGUgIT09IDApIHJlamVjdChuZXcgRXJyb3IoYFdvcmtlciAke2l9IGV4aXRlZCB3aXRoIGNvZGUgJHtjb2RlfWApKTtcbiAgICAgIH0pO1xuICAgIH0pO1xuICB9KTtcblxuICBQcm9taXNlLmFsbChwcm9taXNlcylcbiAgICAudGhlbigocmVzdWx0cykgPT4ge1xuICAgICAgLy8gQ29tYmluZSBhbGwgd29ya2VyIHJlc3VsdHMgaW50byBhIHNpbmdsZSBKU09OIG9iamVjdFxuICAgICAgLy8gU3RydWN0dXJlOiB7IFwicHJvbXB0IHRleHRcIjogW1wicmVzcG9uc2UxXCIsIFwicmVzcG9uc2UyXCIsIFwicmVzcG9uc2UzXCIsIFwicmVzcG9uc2U0XCJdIH1cbiAgICAgIGNvbnN0IGNvbWJpbmVkUmVzdWx0czogUmVjb3JkPHN0cmluZywgc3RyaW5nW10+ID0ge307XG5cbiAgICAgIHJlc3VsdHMuZm9yRWFjaCgod29ya2VyUmVzdWx0KSA9PiB7XG4gICAgICAgIE9iamVjdC5rZXlzKHdvcmtlclJlc3VsdCkuZm9yRWFjaChwcm9tcHQgPT4ge1xuICAgICAgICAgIGNvbWJpbmVkUmVzdWx0c1twcm9tcHRdID0gd29ya2VyUmVzdWx0W3Byb21wdF07XG4gICAgICAgIH0pO1xuICAgICAgfSk7XG5cbiAgICAgIC8vIFdyaXRlIHJlc3VsdHMgdG8gSlNPTiBmaWxlXG4gICAgICBjb25zdCBvdXRwdXRQYXRoID0gJy4vcHJvbXB0X3Jlc3BvbnNlcy5qc29uJztcbiAgICAgIGZzLndyaXRlRmlsZVN5bmMob3V0cHV0UGF0aCwgSlNPTi5zdHJpbmdpZnkoY29tYmluZWRSZXN1bHRzLCBudWxsLCAyKSwgJ3V0ZjgnKTtcbiAgICAgIGNvbnNvbGUubG9nKGBcXG5SZXN1bHRzIHdyaXR0ZW4gdG8gJHtvdXRwdXRQYXRofWApO1xuXG4gICAgICBjb25zb2xlLmxvZygnXFxuPT09IFNVTU1BUlkgPT09Jyk7XG4gICAgICBjb25zb2xlLmxvZyhgVG90YWwgd29ya2VyczogJHtyZXN1bHRzLmxlbmd0aH1gKTtcbiAgICAgIGNvbnNvbGUubG9nKGBUb3RhbCBwcm9tcHRzOiAke09iamVjdC5rZXlzKGNvbWJpbmVkUmVzdWx0cykubGVuZ3RofWApO1xuICAgICAgY29uc29sZS5sb2coYFRvdGFsIHJlc3BvbnNlczogJHtPYmplY3QudmFsdWVzKGNvbWJpbmVkUmVzdWx0cykuZmxhdCgpLmxlbmd0aH1gKTtcbiAgICAgIGNvbnNvbGUubG9nKCc9PT09PT09PT09PT09PT0nKTtcblxuICAgICAgLy8gTG9nIGEgc2FtcGxlIG9mIHJlc3VsdHMgZm9yIHZlcmlmaWNhdGlvblxuICAgICAgT2JqZWN0LmtleXMoY29tYmluZWRSZXN1bHRzKS5mb3JFYWNoKHByb21wdCA9PiB7XG4gICAgICAgIGNvbnNvbGUubG9nKGBcXG5Qcm9tcHQ6IFwiJHtwcm9tcHR9XCJgKTtcbiAgICAgICAgY29uc29sZS5sb2coYFJlc3BvbnNlcyBjb2xsZWN0ZWQ6ICR7Y29tYmluZWRSZXN1bHRzW3Byb21wdF0ubGVuZ3RofWApO1xuICAgICAgICBjb21iaW5lZFJlc3VsdHNbcHJvbXB0XS5mb3JFYWNoKChyZXNwb25zZSwgaW5kZXgpID0+IHtcbiAgICAgICAgICBjb25zb2xlLmxvZyhgICAke2luZGV4ICsgMX06ICR7cmVzcG9uc2Uuc3Vic3RyaW5nKDAsIDEwMCl9JHtyZXNwb25zZS5sZW5ndGggPiAxMDAgPyAnJyA6ICcnfWApO1xuICAgICAgICB9KTtcbiAgICAgIH0pO1xuICAgIH0pXG4gICAgLmNhdGNoKChlcnJvcikgPT4ge1xuICAgICAgY29uc29sZS5lcnJvcignRXJyb3IgaW4gd29ya2VyczonLCBlcnJvcik7XG4gICAgfSk7XG59XG5cbihhc3luYyAoKSA9PiB7XG5cbiAgY29uc3QgY29udGV4dCA9IGF3YWl0IGNocm9taXVtLmxhdW5jaFBlcnNpc3RlbnRDb250ZXh0KGJhc2VfcHJvZmlsZSwge1xuICAgIGNoYW5uZWw6ICdjaHJvbWUnLFxuICAgIGhlYWRsZXNzOiBmYWxzZSxcbiAgICB2aWV3cG9ydDp7aGVpZ2h0OjAsIHdpZHRoOjB9XG4gIH0pO1xuXG4gIGFzeW5jIGZ1bmN0aW9uIHNldHVwRXZlbnRMaXN0ZW5lcnMocGFnZTogUGFnZSkge1xuICAgIGlmICghaGFzUmVnaXN0ZXJlZCkge1xuICAgICAgYXdhaXQgcGFnZS5leHBvc2VGdW5jdGlvbignbG9nRXZlbnQnLCBhc3luYyAodHlwZTogRXZlbnRUeXBlLCBkZXRhaWxzPzogRXZlbnREZXRhaWxzKSA9PiB7XG5cbiAgICAgICAgaWYgKCFkZXRhaWxzKSB7XG4gICAgICAgICAgZGV0YWlscyA9IHsgbWVzc2FnZTogdHlwZSB9XG4gICAgICAgIH1cblxuICAgICAgICBsZXQgbG9nRW50cnk6IEN1c3RvbUV2ZW50TG9nID0geyB0eXBlLCBkZXRhaWxzLCB0aW1lc3RhbXA6IERhdGUubm93KCkgfTtcblxuICAgICAgICBldmVudExvZ3MucHVzaChsb2dFbnRyeSk7XG4gICAgICB9KTtcblxuICAgICAgYXdhaXQgcGFnZS5leHBvc2VGdW5jdGlvbignY2xvc2VCcm93c2VyJywgKCkgPT4ge1xuICAgICAgICBjb250ZXh0LmNsb3NlKCkudGhlbihhc3luYyAoKSA9PiB7XG5cbiAgICAgICAgICBjb25zdCBwcm9maWxlRGlycyA9IGNyZWF0ZVByb2ZpbGVDb3BpZXMoKTtcbiAgICAgICAgICBsZXQgd29ya2VyX2RpcnMgPSBbXVxuICAgICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbnVtX3dvcmtlcnM7IGkrKykge1xuICAgICAgICAgICAgd29ya2VyX2RpcnMucHVzaCgnLi9jbG9uZWQtcHJvZmlsZXMvcHJvZmlsZS0nICsgaS50b1N0cmluZygpKVxuICAgICAgICAgIH1cbiAgICAgICAgICBzcGF3bldvcmtlcnMoYXdhaXQgcHJvZmlsZURpcnMpO1xuICAgICAgICB9KVxuICAgICAgfSlcblxuICAgICAgaGFzUmVnaXN0ZXJlZCA9IHRydWU7XG4gICAgfVxuXG4gICAgYXdhaXQgcGFnZS5ldmFsdWF0ZSgoKSA9PiB7XG4gICAgICBjb25zb2xlLmxvZyhcIkF0dGFjaGluZyBsaXN0ZW5lcnMuLi5cIik7XG5cbiAgICAgIC8vIFJlbW92ZSBvbGQgbGlzdGVuZXJzIGlmIHRoZXkgZXhpc3RcbiAgICAgIGlmICgod2luZG93IGFzIGFueSkuX21vdXNlTW92ZUxpc3RlbmVyKSB7XG4gICAgICAgIGRvY3VtZW50LnJlbW92ZUV2ZW50TGlzdGVuZXIoJ21vdXNlbW92ZScsICh3aW5kb3cgYXMgYW55KS5fbW91c2VNb3ZlTGlzdGVuZXIpO1xuICAgICAgICBkb2N1bWVudC5yZW1vdmVFdmVudExpc3RlbmVyKCdjbGljaycsICh3aW5kb3cgYXMgYW55KS5fY2xpY2tMaXN0ZW5lcik7XG4gICAgICAgIGRvY3VtZW50LnJlbW92ZUV2ZW50TGlzdGVuZXIoJ2RibGNsaWNrJywgKHdpbmRvdyBhcyBhbnkpLl9kYmxDbGlja0xpc3RlbmVyKTtcbiAgICAgICAgZG9jdW1lbnQucmVtb3ZlRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsICh3aW5kb3cgYXMgYW55KS5fa2V5RG93bkxpc3RlbmVyKTtcbiAgICAgIH1cblxuICAgICAgLy8gRGVmaW5lIGxpc3RlbmVyc1xuICAgICAgY29uc3QgbW91c2VNb3ZlTGlzdGVuZXIgPSAoZTogTW91c2VFdmVudCkgPT4ge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIHdpbmRvdy5sb2dFdmVudCgnbW91c2Vtb3ZlJywgeyB4OiBlLmNsaWVudFgsIHk6IGUuY2xpZW50WSB9KTtcbiAgICAgICAgfSBjYXRjaCAoZSkge1xuICAgICAgICAgIGNvbnNvbGUuZXJyb3IoZSlcbiAgICAgICAgICBjb25zb2xlLmxvZyhcIkZBSUxFRFwiKVxuICAgICAgICAgIHdpbmRvdy5sb2NhdGlvbi5yZWxvYWQoKVxuICAgICAgICB9XG4gICAgICB9O1xuXG4gICAgICBjb25zdCBjbGlja0xpc3RlbmVyID0gKGU6IE1vdXNlRXZlbnQpID0+IHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICB3aW5kb3cubG9nRXZlbnQoJ2NsaWNrJywgeyB4OiBlLmNsaWVudFgsIHk6IGUuY2xpZW50WSwgYnV0dG9uOiBlLmJ1dHRvbiB9KTtcbiAgICAgICAgfSBjYXRjaCAoZSkge1xuICAgICAgICAgIGNvbnNvbGUuZXJyb3IoZSlcbiAgICAgICAgICBjb25zb2xlLmxvZyhcIkZBSUxFRFwiKVxuICAgICAgICAgIHdpbmRvdy5sb2NhdGlvbi5yZWxvYWQoKVxuICAgICAgICB9XG4gICAgICB9XG5cblxuICAgICAgY29uc3QgZGJsQ2xpY2tMaXN0ZW5lciA9IChlOiBNb3VzZUV2ZW50KSA9PiB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgd2luZG93LmxvZ0V2ZW50KCdkYmxjbGljaycsIHsgeDogZS5jbGllbnRYLCB5OiBlLmNsaWVudFksIGJ1dHRvbjogZS5idXR0b24gfSk7XG4gICAgICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgICAgICBjb25zb2xlLmVycm9yKGUpXG4gICAgICAgICAgY29uc29sZS5sb2coXCJGQUlMRURcIilcbiAgICAgICAgICB3aW5kb3cubG9jYXRpb24ucmVsb2FkKClcbiAgICAgICAgfVxuXG4gICAgICB9O1xuXG4gICAgICBjb25zdCBjb3B5TGlzdGVuZXIgPSBhc3luYyAoZTogQ2xpcGJvYXJkRXZlbnQpID0+IHtcbiAgICAgICAgd2luZG93LmxvZ0V2ZW50KCdjb3B5JylcbiAgICAgICAgY29uc29sZS5sb2coYXdhaXQgbmF2aWdhdG9yLmNsaXBib2FyZC5yZWFkVGV4dCgpKVxuICAgICAgfVxuICAgICAgY29uc3Qga2V5RG93bkxpc3RlbmVyID0gKGU6IEtleWJvYXJkRXZlbnQpID0+IHtcbiAgICAgICAgaWYgKGUua2V5ID09PSAnRjEyJykge1xuICAgICAgICAgIGUucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICBjb25zb2xlLmxvZygnRjEyIHByZXNzZWQgLSBlbnRlcmluZyByZXBsYXkgbW9kZScpO1xuICAgICAgICAgIHdpbmRvdy5yZXBsYXlNb2RlID0gdHJ1ZTtcbiAgICAgICAgICB3aW5kb3cubG9nRXZlbnQoJ2YxMl9wcmVzc2VkJywge1xuICAgICAgICAgICAgbWVzc2FnZTogJ0VudGVyZWQgcmVwbGF5IG1vZGUgLSBwcmVzcyBhbnkga2V5IHRvIHN0YXJ0IHJlcGxheScsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gZWxzZSBpZiAoZS5rZXkgPT09ICdFc2NhcGUnKSB7XG4gICAgICAgICAgd2luZG93LmxvZ0V2ZW50KCdlc2NhcGVfcHJlc3NlZCcsIHsgbWVzc2FnZTogJ0Nsb3NpbmcgYnJvd3NlcicgfSk7XG4gICAgICAgIH0gZWxzZSBpZiAoZS5rZXkgPT09ICdFbnRlcicpIHtcbiAgICAgICAgICB3aW5kb3cubG9nRXZlbnQoJ2VudGVyX2tleV9wcmVzcycsIHsga2V5OiBlLmtleSB9KTtcbiAgICAgICAgfSBlbHNlIGlmICh3aW5kb3cucmVwbGF5TW9kZSkge1xuICAgICAgICAgIGUucHJldmVudERlZmF1bHQoKTtcbiAgICAgICAgICB3aW5kb3cubG9nRXZlbnQoJ3JlcGxheV90cmlnZ2VyJywge1xuICAgICAgICAgICAga2V5OiBlLmtleSxcbiAgICAgICAgICAgIG1lc3NhZ2U6ICdTdGFydGluZyByZXBsYXkuLi4nLFxuICAgICAgICAgIH0pO1xuICAgICAgICAgIHRyeSB7XG4gICAgICAgICAgICB3aW5kb3cuY2xvc2VCcm93c2VyKCk7XG4gICAgICAgICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZyhlcnIpO1xuICAgICAgICAgIH1cbiAgICAgICAgICB3aW5kb3cucmVwbGF5TW9kZSA9IGZhbHNlO1xuICAgICAgICB9XG4gICAgICB9O1xuXG4gICAgICAvLyBTYXZlIHJlZmVyZW5jZXMgZ2xvYmFsbHkgdG8gYWxsb3cgZnV0dXJlIHJlbW92YWxcbiAgICAgICh3aW5kb3cgYXMgYW55KS5fbW91c2VNb3ZlTGlzdGVuZXIgPSBtb3VzZU1vdmVMaXN0ZW5lcjtcbiAgICAgICh3aW5kb3cgYXMgYW55KS5fY2xpY2tMaXN0ZW5lciA9IGNsaWNrTGlzdGVuZXI7XG4gICAgICAod2luZG93IGFzIGFueSkuX2RibENsaWNrTGlzdGVuZXIgPSBkYmxDbGlja0xpc3RlbmVyO1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLl9rZXlEb3duTGlzdGVuZXIgPSBrZXlEb3duTGlzdGVuZXI7XG5cbiAgICAgIC8vIEF0dGFjaCBuZXcgbGlzdGVuZXJzXG4gICAgICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZW1vdmUnLCBtb3VzZU1vdmVMaXN0ZW5lcik7XG4gICAgICBkb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIGNsaWNrTGlzdGVuZXIpO1xuICAgICAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignZGJsY2xpY2snLCBkYmxDbGlja0xpc3RlbmVyKTtcbiAgICAgIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2tleWRvd24nLCBrZXlEb3duTGlzdGVuZXIpO1xuICAgICAgZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY29weScsIGNvcHlMaXN0ZW5lcilcbiAgICB9KTtcbiAgfVxuXG4gIGNvbnN0IHBhZ2UgPSBhd2FpdCBjb250ZXh0Lm5ld1BhZ2UoKTtcblxuICBwYWdlLm9uKCdmcmFtZW5hdmlnYXRlZCcsIGFzeW5jICgpID0+IHtcbiAgICB0cnkge1xuICAgICAgYXdhaXQgc2V0dXBFdmVudExpc3RlbmVycyhwYWdlKTtcbiAgICB9XG4gICAgY2F0Y2ggKF8pIHtcbiAgICAgIDtcbiAgICB9XG5cbiAgfSk7XG5cbiAgYXdhaXQgcGFnZS5nb3RvKHdlYnNpdGUpO1xuXG4gIGF3YWl0IHBhZ2Uud2FpdEZvckxvYWRTdGF0ZSgnbG9hZCcpO1xuICBsZXQgZG9uZVJlbG9hZCA9IGZhbHNlXG5cbiAgY29uc29sZS5sb2coJ0xvYWRpbmcgc2NyaXB0Li4uJylcbiAgd2hpbGUoIWRvbmVSZWxvYWQpe1xuICAgIHRyeXtcbiAgICAgICAgYXdhaXQgcGFnZS5yZWxvYWQoKTtcbiAgICAgICAgZG9uZVJlbG9hZCA9IHRydWVcbiAgICB9IGNhdGNoKF8pe1xuICAgICAgO1xuICAgIH1cbiAgfVxuXG4gIGF3YWl0IHBhZ2Uuc2V0Vmlld3BvcnRTaXplKHsgd2lkdGg6IDEyODAsIGhlaWdodDogNzIwIH0pO1xuICBjb25zb2xlLmxvZygnQnJvd3NlciBvcGVuZWQuIEludGVyYWN0IHdpdGggdGhlIHBhZ2U6Jyk7XG4gIGNvbnNvbGUubG9nKCctIFJlY29yZCB5b3VyIGludGVyYWN0aW9ucyBub3JtYWxseScpO1xuICBjb25zb2xlLmxvZygnLSBQcmVzcyBGMTIgdG8gZW50ZXIgcmVwbGF5IG1vZGUnKTtcbiAgY29uc29sZS5sb2coJy0gUHJlc3MgYW55IGtleSBhZnRlciBGMTIgdG8gc3RhcnQgd29ya2VycycpO1xuICBjb25zb2xlLmxvZyhgLSBFYWNoIHdvcmtlciB3aWxsIHJ1biAyIHByb21wdHMsIDQgdGltZXMgZWFjaGApO1xuICBjb25zb2xlLmxvZyhgLSBUb3RhbDogJHtudW1fd29ya2Vyc30gd29ya2VycyDDlyAyIHByb21wdHMgw5cgNCBpdGVyYXRpb25zID0gJHtudW1fd29ya2VycyAqIDIgKiA0fSByZXNwb25zZXNgKTtcbn0pKCkiXSwibmFtZXMiOlsiY2hyb21pdW0iLCJwYXRoIiwiZnMiLCJXb3JrZXIiLCJhcmdzIiwicHJvY2VzcyIsImFyZ3YiLCJzbGljZSIsIndlYnNpdGUiLCJudW1fd29ya2VycyIsInBhcnNlSW50IiwiYmFzZV9wcm9maWxlIiwicHJvZmlsZXNfY2xvbmVzX2RpciIsImV2ZW50TG9ncyIsInByb21wdEFyclR4dCIsInJlYWRGaWxlU3luYyIsIkFMTF9QUk9NUFRTIiwiSlNPTiIsInBhcnNlIiwiY29uc29sZSIsImxvZyIsIndhaXREdXJhdGlvbiIsImhhc1JlZ2lzdGVyZWQiLCJjcmVhdGVQcm9maWxlQ29waWVzIiwiZXhpc3RzU3luYyIsIm1rZGlyU3luYyIsInJlY3Vyc2l2ZSIsImRpcnMiLCJpIiwiY2xvbmVEaXIiLCJqb2luIiwicHVzaCIsIlByb21pc2UiLCJhbGwiLCJtYXAiLCJybSIsImNwIiwic3Bhd25Xb3JrZXJzIiwicHJvZmlsZURpcnMiLCJwcm9tcHRMaXN0IiwiXyIsImxlbmd0aCIsInByb21pc2VzIiwicHJvZmlsZURpciIsInJlc29sdmUiLCJyZWplY3QiLCJ3b3JrZXJQcm9tcHRzIiwid29ya2VyIiwid29ya2VyRGF0YSIsInVzZXJEYXRhRGlyIiwid29ya2VySWQiLCJwcm9tcHRzIiwib24iLCJtc2ciLCJPYmplY3QiLCJrZXlzIiwiY29kZSIsIkVycm9yIiwidGhlbiIsInJlc3VsdHMiLCJjb21iaW5lZFJlc3VsdHMiLCJmb3JFYWNoIiwid29ya2VyUmVzdWx0IiwicHJvbXB0Iiwib3V0cHV0UGF0aCIsIndyaXRlRmlsZVN5bmMiLCJzdHJpbmdpZnkiLCJ2YWx1ZXMiLCJmbGF0IiwicmVzcG9uc2UiLCJpbmRleCIsInN1YnN0cmluZyIsImNhdGNoIiwiZXJyb3IiLCJjb250ZXh0IiwibGF1bmNoUGVyc2lzdGVudENvbnRleHQiLCJjaGFubmVsIiwiaGVhZGxlc3MiLCJ2aWV3cG9ydCIsImhlaWdodCIsIndpZHRoIiwic2V0dXBFdmVudExpc3RlbmVycyIsInBhZ2UiLCJleHBvc2VGdW5jdGlvbiIsInR5cGUiLCJkZXRhaWxzIiwibWVzc2FnZSIsImxvZ0VudHJ5IiwidGltZXN0YW1wIiwiRGF0ZSIsIm5vdyIsImNsb3NlIiwid29ya2VyX2RpcnMiLCJ0b1N0cmluZyIsImV2YWx1YXRlIiwid2luZG93IiwiX21vdXNlTW92ZUxpc3RlbmVyIiwiZG9jdW1lbnQiLCJyZW1vdmVFdmVudExpc3RlbmVyIiwiX2NsaWNrTGlzdGVuZXIiLCJfZGJsQ2xpY2tMaXN0ZW5lciIsIl9rZXlEb3duTGlzdGVuZXIiLCJtb3VzZU1vdmVMaXN0ZW5lciIsImUiLCJsb2dFdmVudCIsIngiLCJjbGllbnRYIiwieSIsImNsaWVudFkiLCJsb2NhdGlvbiIsInJlbG9hZCIsImNsaWNrTGlzdGVuZXIiLCJidXR0b24iLCJkYmxDbGlja0xpc3RlbmVyIiwiY29weUxpc3RlbmVyIiwibmF2aWdhdG9yIiwiY2xpcGJvYXJkIiwicmVhZFRleHQiLCJrZXlEb3duTGlzdGVuZXIiLCJrZXkiLCJwcmV2ZW50RGVmYXVsdCIsInJlcGxheU1vZGUiLCJjbG9zZUJyb3dzZXIiLCJlcnIiLCJhZGRFdmVudExpc3RlbmVyIiwibmV3UGFnZSIsImdvdG8iLCJ3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