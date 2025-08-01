import { parentPort, workerData } from 'worker_threads';
import { chromium } from 'patchright';
// sanitize.js
import { JSDOM } from 'jsdom';
import createDOMPurify from 'dompurify';
import { writeFileSync } from 'fs'; // named import
import dotenv from "dotenv";
import OpenAI from 'openai';
dotenv.config(); // load .env into process.env
// Create a virtual DOM window
const eventLogs = workerData.eventLogs;
const userDataDir = workerData.userDataDir;
const workerId = workerData.workerId;
const prompts = workerData.prompts;
const website = workerData.website;
const waitDuration = workerData.waitDuration;
const ITERATIONS_PER_PROMPT = 1;
async function replayActionsAndGetResponse(page, logs, promptText) {
    try {
        const actionsToReplay = logs.filter((event)=>![
                'f12_pressed',
                'escape_pressed',
                'replay_trigger'
            ].includes(event.type)).sort((a, b)=>a.timestamp - b.timestamp);
        console.log(`[Worker ${workerId}] Replaying ${actionsToReplay.length} actions for prompt: "${promptText}"`);
        let lastTimestamp = actionsToReplay[0]?.timestamp ?? Date.now();
        for (const action of actionsToReplay){
            try {
                const delay = action.timestamp - lastTimestamp;
                if (delay > 0) {
                    await new Promise((res)=>setTimeout(res, Math.min(delay, 1000))); // Cap delay at 1 second
                }
                lastTimestamp = action.timestamp;
                switch(action.type){
                    case 'mousemove':
                        {
                            const { x, y } = action.details;
                            await page.mouse.move(x, y);
                            console.log(await page.evaluate("window.getSelection().toString();"));
                            break;
                        }
                    case 'mousedown':
                        {
                            await page.mouse.down();
                        }
                    case 'click':
                        {
                            const { x, y, button } = action.details;
                            await page.mouse.click(x, y, {
                                button: button === 0 ? 'left' : button === 1 ? 'middle' : 'right'
                            });
                            break;
                        }
                    case 'dblclick':
                        {
                            const { x, y, button } = action.details;
                            const clickOpts = {
                                button: button === 0 ? 'left' : button === 1 ? 'middle' : 'right'
                            };
                            await page.mouse.dblclick(x, y, clickOpts);
                            break;
                        }
                    case 'enter_key_press':
                        {
                            // Type our custom prompt text instead of default text
                            await page.keyboard.type(promptText);
                            await page.keyboard.press('Enter');
                            break;
                        }
                    case 'keydown':
                        {
                            const { key } = action.details;
                            await page.keyboard.press(key);
                            break;
                        }
                    case 'copy':
                        {
                            console.log(await page.evaluate("window.getSelection().toString();"));
                            await page.keyboard.press("Control+C");
                            console.log("hello");
                            await page.evaluate(async ()=>{
                                console.log(await navigator.clipboard.readText());
                            });
                            console.log("hello");
                        }
                }
            } catch (err) {
                console.error(`[Worker ${workerId}] Error during ${action.type}:`, err.message);
            }
        }
        // Wait for ChatGPT to respond
        console.log(`[Worker ${workerId}] Waiting for response to: "${promptText}"`);
        await page.waitForTimeout(waitDuration); // Wait longer for AI response
        let content = await page.content();
        const window = new JSDOM('').window;
        const DOMPurify = createDOMPurify(window);
        const clean = DOMPurify.sanitize(content, {
            ALLOWED_ATTR: [],
            ALLOW_DATA_ATTR: false,
            ALLOW_ARIA_ATTR: false
        });
        writeFileSync(Date.now() + ".html", clean);
        // Get selected text using the same method as the original code
        const selectedText = await page.evaluate(()=>{
            const selection = window.getSelection();
            return selection ? selection.toString().trim() : '';
        });
        console.log(`[Worker ${workerId}] Selected text after replay: "${selectedText.substring(0, 100)}"`);
        const openai = new OpenAI({
            apiKey: process.env.DEEPINFRA_API_KEY,
            baseURL: "https://api.deepinfra.com/v1/openai"
        });
        const reply = await openai.chat.completions.create({
            model: "Qwen/Qwen3-235B-A22B",
            temperature: 0.0,
            messages: [
                {
                    role: "user",
                    content: `Extract the response of the LLM from the following conversation, which is formatted in HTML. Answer with ONLY the response of the LLM, nothing else. In your answer, make sure to include "prefix phrases" such as "sure, let me help you with that", and "suffix phrases" such as "can I help you with anything else" if the LLM in the conversation responded with those phrases. Include LLM response text in your answer only if it is meant to the visible to the end user. Here is the conversation: ${clean}`
                }
            ],
            stream: false
        });
        return reply.choices[0].message.content?.split("</think>").at(-1);
    } catch (error) {
        console.error(`[Worker ${workerId}] Error replaying actions for "${promptText}":`, error.message);
        return `Error: ${error.message}`;
    }
}
async function runPromptIterations() {
    const results = {};
    // Initialize results structure for both prompts
    prompts.forEach((prompt)=>{
        results[prompt] = [];
    });
    // Run each prompt 4 times
    for (const prompt of prompts){
        console.log(`[Worker ${workerId}] Starting iterations for prompt: "${prompt}"`);
        for(let iteration = 1; iteration <= ITERATIONS_PER_PROMPT; iteration++){
            console.log(`[Worker ${workerId}] Running iteration ${iteration}/${ITERATIONS_PER_PROMPT} for: "${prompt}"`);
            try {
                // Launch new browser context for each iteration
                const context = await chromium.launchPersistentContext(userDataDir, {
                    channel: 'chrome',
                    headless: false
                });
                const page = await context.newPage();
                await page.goto(website);
                await page.waitForLoadState('load');
                let doneReload = false;
                while(!doneReload){
                    try {
                        await page.reload();
                        doneReload = true;
                    } catch (_) {
                        ;
                    }
                }
                await page.evaluate(()=>{
                    document.addEventListener('mousemove', (e)=>{
                        // Create a red dot element
                        console.log("hi");
                        const dot = document.createElement('div');
                        dot.style.position = 'absolute';
                        dot.style.width = '10px';
                        dot.style.height = '10px';
                        dot.style.backgroundColor = 'red';
                        dot.style.borderRadius = '50%';
                        dot.style.left = `${e.clientX}px`;
                        dot.style.top = `${e.clientY}px`;
                        dot.style.pointerEvents = 'none';
                        dot.style.zIndex = '9999';
                        document.body.appendChild(dot);
                        // Optional: remove the dot after a short time
                        setTimeout(()=>dot.remove(), 500);
                    });
                });
                // Replay user actions and get response
                const response = await replayActionsAndGetResponse(page, eventLogs, prompt);
                // Store the response
                results[prompt].push(response);
                // Close browser before next iteration
                await context.close();
                console.log(`[Worker ${workerId}] Completed iteration ${iteration} for: "${prompt}"`);
                // Wait between iterations to avoid rate limiting
                await new Promise((resolve)=>setTimeout(resolve, 2000));
            } catch (error) {
                console.error(`[Worker ${workerId}] Error in iteration ${iteration} for "${prompt}":`, error.message);
                results[prompt].push(`Error in iteration ${iteration}: ${error.message}`);
            }
        }
        console.log(`[Worker ${workerId}] Completed all ${ITERATIONS_PER_PROMPT} iterations for: "${prompt}"`);
        // Longer wait between different prompts
        if (prompts.indexOf(prompt) < prompts.length - 1) {
            await new Promise((resolve)=>setTimeout(resolve, 3000));
        }
    }
    return results;
}
(async ()=>{
    console.log(`[Worker ${workerId}] Starting with userDataDir: ${userDataDir}`);
    console.log(`[Worker ${workerId}] Assigned prompts:`, prompts);
    console.log(`[Worker ${workerId}] Will run each prompt ${ITERATIONS_PER_PROMPT} times`);
    // Clean up any existing browser instances
    try {
        const preContext = await chromium.launchPersistentContext(userDataDir, {
            channel: 'chrome',
            headless: true
        });
        await preContext.close();
    } catch (error) {
        console.log(`[Worker ${workerId}] No existing context to clean up`);
    }
    const results = await runPromptIterations();
    // Log summary for this worker
    console.log(`[Worker ${workerId}] COMPLETED ALL WORK!`);
    console.log(`[Worker ${workerId}] Summary:`);
    Object.keys(results).forEach((prompt)=>{
        console.log(`  - "${prompt}": ${results[prompt].length} responses collected`);
    });
    parentPort?.postMessage(results);
})();

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIi4uLy4uL3NyYy93b3JrZXIudHMiXSwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgcGFyZW50UG9ydCwgd29ya2VyRGF0YSB9IGZyb20gJ3dvcmtlcl90aHJlYWRzJztcclxuaW1wb3J0IHsgY2hyb21pdW0sIFBhZ2UsIEJyb3dzZXJDb250ZXh0IH0gZnJvbSAncGF0Y2hyaWdodCc7XHJcbi8vIHNhbml0aXplLmpzXHJcbmltcG9ydCB7IEpTRE9NIH0gZnJvbSAnanNkb20nO1xyXG5pbXBvcnQgY3JlYXRlRE9NUHVyaWZ5IGZyb20gJ2RvbXB1cmlmeSc7XHJcbmltcG9ydCB7IHdyaXRlRmlsZVN5bmMgfSBmcm9tICdmcyc7IC8vIG5hbWVkIGltcG9ydFxyXG5pbXBvcnQgZG90ZW52IGZyb20gXCJkb3RlbnZcIjtcclxuaW1wb3J0IE9wZW5BSSBmcm9tICdvcGVuYWknO1xyXG5kb3RlbnYuY29uZmlnKCk7ICAvLyBsb2FkIC5lbnYgaW50byBwcm9jZXNzLmVudlxyXG4vLyBDcmVhdGUgYSB2aXJ0dWFsIERPTSB3aW5kb3dcclxuXHJcblxyXG5jb25zdCBldmVudExvZ3M6IEN1c3RvbUV2ZW50TG9nW10gPSB3b3JrZXJEYXRhLmV2ZW50TG9ncztcclxuY29uc3QgdXNlckRhdGFEaXI6IHN0cmluZyA9IHdvcmtlckRhdGEudXNlckRhdGFEaXI7XHJcbmNvbnN0IHdvcmtlcklkOiBudW1iZXIgPSB3b3JrZXJEYXRhLndvcmtlcklkO1xyXG5jb25zdCBwcm9tcHRzOiBzdHJpbmdbXSA9IHdvcmtlckRhdGEucHJvbXB0cztcclxuY29uc3Qgd2Vic2l0ZTogc3RyaW5nID0gd29ya2VyRGF0YS53ZWJzaXRlXHJcbmNvbnN0IHdhaXREdXJhdGlvbjogbnVtYmVyID0gd29ya2VyRGF0YS53YWl0RHVyYXRpb25cclxuY29uc3QgSVRFUkFUSU9OU19QRVJfUFJPTVBUID0gMTtcclxuXHJcbmFzeW5jIGZ1bmN0aW9uIHJlcGxheUFjdGlvbnNBbmRHZXRSZXNwb25zZShwYWdlOiBQYWdlLCBsb2dzOiBDdXN0b21FdmVudExvZ1tdLCBwcm9tcHRUZXh0OiBzdHJpbmcpOiBQcm9taXNlPHN0cmluZz4ge1xyXG4gIHRyeSB7XHJcbiAgICBjb25zdCBhY3Rpb25zVG9SZXBsYXkgPSBsb2dzXHJcbiAgICAgIC5maWx0ZXIoKGV2ZW50KSA9PiAhWydmMTJfcHJlc3NlZCcsICdlc2NhcGVfcHJlc3NlZCcsICdyZXBsYXlfdHJpZ2dlciddLmluY2x1ZGVzKGV2ZW50LnR5cGUpKVxyXG4gICAgICAuc29ydCgoYSwgYikgPT4gYS50aW1lc3RhbXAgLSBiLnRpbWVzdGFtcCk7XHJcblxyXG4gICAgY29uc29sZS5sb2coYFtXb3JrZXIgJHt3b3JrZXJJZH1dIFJlcGxheWluZyAke2FjdGlvbnNUb1JlcGxheS5sZW5ndGh9IGFjdGlvbnMgZm9yIHByb21wdDogXCIke3Byb21wdFRleHR9XCJgKTtcclxuXHJcbiAgICBsZXQgbGFzdFRpbWVzdGFtcCA9IGFjdGlvbnNUb1JlcGxheVswXT8udGltZXN0YW1wID8/IERhdGUubm93KCk7XHJcblxyXG4gICAgZm9yIChjb25zdCBhY3Rpb24gb2YgYWN0aW9uc1RvUmVwbGF5KSB7XHJcbiAgICAgIHRyeSB7XHJcbiAgICAgICAgY29uc3QgZGVsYXkgPSBhY3Rpb24udGltZXN0YW1wIC0gbGFzdFRpbWVzdGFtcDtcclxuICAgICAgICBpZiAoZGVsYXkgPiAwKSB7XHJcbiAgICAgICAgICBhd2FpdCBuZXcgUHJvbWlzZSgocmVzKSA9PiBzZXRUaW1lb3V0KHJlcywgTWF0aC5taW4oZGVsYXksIDEwMDApKSk7IC8vIENhcCBkZWxheSBhdCAxIHNlY29uZFxyXG4gICAgICAgIH1cclxuICAgICAgICBsYXN0VGltZXN0YW1wID0gYWN0aW9uLnRpbWVzdGFtcDtcclxuXHJcbiAgICAgICAgc3dpdGNoIChhY3Rpb24udHlwZSkge1xyXG4gICAgICAgICAgY2FzZSAnbW91c2Vtb3ZlJzoge1xyXG4gICAgICAgICAgICBjb25zdCB7IHgsIHkgfSA9IGFjdGlvbi5kZXRhaWxzIGFzIHsgeDogbnVtYmVyOyB5OiBudW1iZXIgfTtcclxuICAgICAgICAgICAgYXdhaXQgcGFnZS5tb3VzZS5tb3ZlKHgsIHkpO1xyXG4gICAgICAgICAgICBjb25zb2xlLmxvZyhhd2FpdCBwYWdlLmV2YWx1YXRlKFwid2luZG93LmdldFNlbGVjdGlvbigpLnRvU3RyaW5nKCk7XCIpKVxyXG4gICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgIH1cclxuICAgICAgICAgIGNhc2UgJ21vdXNlZG93bic6IHtcclxuICAgICAgICAgICAgYXdhaXQgcGFnZS5tb3VzZS5kb3duKClcclxuXHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgICBjYXNlICdjbGljayc6IHtcclxuICAgICAgICAgICAgY29uc3QgeyB4LCB5LCBidXR0b24gfSA9IGFjdGlvbi5kZXRhaWxzIGFzIHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IGJ1dHRvbjogbnVtYmVyIH07XHJcbiAgICAgICAgICAgIGF3YWl0IHBhZ2UubW91c2UuY2xpY2soeCwgeSwge1xyXG4gICAgICAgICAgICAgIGJ1dHRvbjogYnV0dG9uID09PSAwID8gJ2xlZnQnIDogYnV0dG9uID09PSAxID8gJ21pZGRsZScgOiAncmlnaHQnLFxyXG4gICAgICAgICAgICB9KTtcclxuICAgICAgICAgICAgYnJlYWs7XHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgICBjYXNlICdkYmxjbGljayc6IHtcclxuICAgICAgICAgICAgY29uc3QgeyB4LCB5LCBidXR0b24gfSA9IGFjdGlvbi5kZXRhaWxzIGFzIHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IGJ1dHRvbjogbnVtYmVyIH07XHJcbiAgICAgICAgICAgIGNvbnN0IGNsaWNrT3B0czogeyBidXR0b24/OiBcImxlZnRcIiB8IFwibWlkZGxlXCIgfCBcInJpZ2h0XCIgfCB1bmRlZmluZWQgfSA9IHtcclxuICAgICAgICAgICAgICBidXR0b246IGJ1dHRvbiA9PT0gMCA/ICdsZWZ0JyA6IGJ1dHRvbiA9PT0gMSA/ICdtaWRkbGUnIDogJ3JpZ2h0JyxcclxuICAgICAgICAgICAgfTtcclxuICAgICAgICAgICAgYXdhaXQgcGFnZS5tb3VzZS5kYmxjbGljayh4LCB5LCBjbGlja09wdHMpO1xyXG4gICAgICAgICAgICBicmVhaztcclxuICAgICAgICAgIH1cclxuICAgICAgICAgIGNhc2UgJ2VudGVyX2tleV9wcmVzcyc6IHtcclxuICAgICAgICAgICAgLy8gVHlwZSBvdXIgY3VzdG9tIHByb21wdCB0ZXh0IGluc3RlYWQgb2YgZGVmYXVsdCB0ZXh0XHJcbiAgICAgICAgICAgIGF3YWl0IHBhZ2Uua2V5Ym9hcmQudHlwZShwcm9tcHRUZXh0KTtcclxuICAgICAgICAgICAgYXdhaXQgcGFnZS5rZXlib2FyZC5wcmVzcygnRW50ZXInKTtcclxuICAgICAgICAgICAgYnJlYWs7XHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgICBjYXNlICdrZXlkb3duJzoge1xyXG4gICAgICAgICAgICBjb25zdCB7IGtleSB9ID0gYWN0aW9uLmRldGFpbHMgYXMgeyBrZXk6IHN0cmluZyB9O1xyXG4gICAgICAgICAgICBhd2FpdCBwYWdlLmtleWJvYXJkLnByZXNzKGtleSk7XHJcbiAgICAgICAgICAgIGJyZWFrO1xyXG4gICAgICAgICAgfVxyXG5cclxuICAgICAgICAgIGNhc2UgJ2NvcHknOiB7XHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKGF3YWl0IHBhZ2UuZXZhbHVhdGUoXCJ3aW5kb3cuZ2V0U2VsZWN0aW9uKCkudG9TdHJpbmcoKTtcIikpXHJcbiAgICAgICAgICAgIGF3YWl0IHBhZ2Uua2V5Ym9hcmQucHJlc3MoXCJDb250cm9sK0NcIilcclxuICAgICAgICAgICAgY29uc29sZS5sb2coXCJoZWxsb1wiKVxyXG4gICAgICAgICAgICBhd2FpdCBwYWdlLmV2YWx1YXRlKGFzeW5jICgpID0+IHtcclxuICAgICAgICAgICAgICBjb25zb2xlLmxvZyhhd2FpdCBuYXZpZ2F0b3IuY2xpcGJvYXJkLnJlYWRUZXh0KCkpXHJcbiAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKFwiaGVsbG9cIilcclxuICAgICAgICAgIH1cclxuXHJcbiAgICAgICAgfVxyXG4gICAgICB9IGNhdGNoIChlcnIpIHtcclxuICAgICAgICBjb25zb2xlLmVycm9yKGBbV29ya2VyICR7d29ya2VySWR9XSBFcnJvciBkdXJpbmcgJHthY3Rpb24udHlwZX06YCwgKGVyciBhcyBFcnJvcikubWVzc2FnZSk7XHJcbiAgICAgIH1cclxuICAgIH1cclxuXHJcbiAgICAvLyBXYWl0IGZvciBDaGF0R1BUIHRvIHJlc3BvbmRcclxuICAgIGNvbnNvbGUubG9nKGBbV29ya2VyICR7d29ya2VySWR9XSBXYWl0aW5nIGZvciByZXNwb25zZSB0bzogXCIke3Byb21wdFRleHR9XCJgKTtcclxuXHJcbiAgICBhd2FpdCBwYWdlLndhaXRGb3JUaW1lb3V0KHdhaXREdXJhdGlvbik7IC8vIFdhaXQgbG9uZ2VyIGZvciBBSSByZXNwb25zZVxyXG4gICAgXHJcbiAgICBsZXQgY29udGVudCA9IGF3YWl0IHBhZ2UuY29udGVudCgpXHJcbiAgICBjb25zdCB3aW5kb3cgPSBuZXcgSlNET00oJycpLndpbmRvdztcclxuICAgIGNvbnN0IERPTVB1cmlmeSA9IGNyZWF0ZURPTVB1cmlmeSh3aW5kb3cpO1xyXG5cclxuICAgIGNvbnN0IGNsZWFuID0gRE9NUHVyaWZ5LnNhbml0aXplKGNvbnRlbnQsIHsgQUxMT1dFRF9BVFRSOiBbXSwgQUxMT1dfREFUQV9BVFRSOiBmYWxzZSwgQUxMT1dfQVJJQV9BVFRSOiBmYWxzZSB9KTtcclxuXHJcbiAgICB3cml0ZUZpbGVTeW5jKERhdGUubm93KCkgKyBcIi5odG1sXCIsIGNsZWFuKVxyXG4gICAgLy8gR2V0IHNlbGVjdGVkIHRleHQgdXNpbmcgdGhlIHNhbWUgbWV0aG9kIGFzIHRoZSBvcmlnaW5hbCBjb2RlXHJcbiAgICBjb25zdCBzZWxlY3RlZFRleHQgPSBhd2FpdCBwYWdlLmV2YWx1YXRlKCgpID0+IHtcclxuICAgICAgY29uc3Qgc2VsZWN0aW9uID0gd2luZG93LmdldFNlbGVjdGlvbigpO1xyXG4gICAgICByZXR1cm4gc2VsZWN0aW9uID8gc2VsZWN0aW9uLnRvU3RyaW5nKCkudHJpbSgpIDogJyc7XHJcbiAgICB9KTtcclxuXHJcbiAgICBjb25zb2xlLmxvZyhgW1dvcmtlciAke3dvcmtlcklkfV0gU2VsZWN0ZWQgdGV4dCBhZnRlciByZXBsYXk6IFwiJHtzZWxlY3RlZFRleHQuc3Vic3RyaW5nKDAsIDEwMCl9XCJgKTtcclxuXHJcbiAgICBjb25zdCBvcGVuYWkgPSBuZXcgT3BlbkFJKHtcclxuICAgICAgYXBpS2V5OiBwcm9jZXNzLmVudi5ERUVQSU5GUkFfQVBJX0tFWSxcclxuICAgICAgYmFzZVVSTDogXCJodHRwczovL2FwaS5kZWVwaW5mcmEuY29tL3YxL29wZW5haVwiLFxyXG4gICAgfSk7XHJcblxyXG4gICAgY29uc3QgcmVwbHkgPSBhd2FpdCBvcGVuYWkuY2hhdC5jb21wbGV0aW9ucy5jcmVhdGUoe1xyXG4gICAgICBtb2RlbDogXCJRd2VuL1F3ZW4zLTIzNUItQTIyQlwiLFxyXG4gICAgICB0ZW1wZXJhdHVyZTowLjAsXHJcbiAgICAgIG1lc3NhZ2VzOiBbXHJcbiAgICAgICAgeyByb2xlOiBcInVzZXJcIiwgY29udGVudDogYEV4dHJhY3QgdGhlIHJlc3BvbnNlIG9mIHRoZSBMTE0gZnJvbSB0aGUgZm9sbG93aW5nIGNvbnZlcnNhdGlvbiwgd2hpY2ggaXMgZm9ybWF0dGVkIGluIEhUTUwuIEFuc3dlciB3aXRoIE9OTFkgdGhlIHJlc3BvbnNlIG9mIHRoZSBMTE0sIG5vdGhpbmcgZWxzZS4gSW4geW91ciBhbnN3ZXIsIG1ha2Ugc3VyZSB0byBpbmNsdWRlIFwicHJlZml4IHBocmFzZXNcIiBzdWNoIGFzIFwic3VyZSwgbGV0IG1lIGhlbHAgeW91IHdpdGggdGhhdFwiLCBhbmQgXCJzdWZmaXggcGhyYXNlc1wiIHN1Y2ggYXMgXCJjYW4gSSBoZWxwIHlvdSB3aXRoIGFueXRoaW5nIGVsc2VcIiBpZiB0aGUgTExNIGluIHRoZSBjb252ZXJzYXRpb24gcmVzcG9uZGVkIHdpdGggdGhvc2UgcGhyYXNlcy4gSW5jbHVkZSBMTE0gcmVzcG9uc2UgdGV4dCBpbiB5b3VyIGFuc3dlciBvbmx5IGlmIGl0IGlzIG1lYW50IHRvIHRoZSB2aXNpYmxlIHRvIHRoZSBlbmQgdXNlci4gSGVyZSBpcyB0aGUgY29udmVyc2F0aW9uOiAke2NsZWFufWAgfVxyXG4gICAgICBdLFxyXG4gICAgICBzdHJlYW06IGZhbHNlLFxyXG4gICAgfSk7XHJcblxyXG5cclxuICAgIHJldHVybiByZXBseS5jaG9pY2VzWzBdLm1lc3NhZ2UuY29udGVudD8uc3BsaXQoXCI8L3RoaW5rPlwiKS5hdCgtMSkgYXMgc3RyaW5nXHJcblxyXG4gIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICBjb25zb2xlLmVycm9yKGBbV29ya2VyICR7d29ya2VySWR9XSBFcnJvciByZXBsYXlpbmcgYWN0aW9ucyBmb3IgXCIke3Byb21wdFRleHR9XCI6YCwgKGVycm9yIGFzIEVycm9yKS5tZXNzYWdlKTtcclxuICAgIHJldHVybiBgRXJyb3I6ICR7KGVycm9yIGFzIEVycm9yKS5tZXNzYWdlfWA7XHJcbiAgfVxyXG59XHJcblxyXG5hc3luYyBmdW5jdGlvbiBydW5Qcm9tcHRJdGVyYXRpb25zKCk6IFByb21pc2U8UmVjb3JkPHN0cmluZywgc3RyaW5nW10+PiB7XHJcbiAgY29uc3QgcmVzdWx0czogUmVjb3JkPHN0cmluZywgc3RyaW5nW10+ID0ge307XHJcblxyXG4gIC8vIEluaXRpYWxpemUgcmVzdWx0cyBzdHJ1Y3R1cmUgZm9yIGJvdGggcHJvbXB0c1xyXG4gIHByb21wdHMuZm9yRWFjaChwcm9tcHQgPT4ge1xyXG4gICAgcmVzdWx0c1twcm9tcHRdID0gW107XHJcbiAgfSk7XHJcblxyXG4gIC8vIFJ1biBlYWNoIHByb21wdCA0IHRpbWVzXHJcbiAgZm9yIChjb25zdCBwcm9tcHQgb2YgcHJvbXB0cykge1xyXG4gICAgY29uc29sZS5sb2coYFtXb3JrZXIgJHt3b3JrZXJJZH1dIFN0YXJ0aW5nIGl0ZXJhdGlvbnMgZm9yIHByb21wdDogXCIke3Byb21wdH1cImApO1xyXG5cclxuICAgIGZvciAobGV0IGl0ZXJhdGlvbiA9IDE7IGl0ZXJhdGlvbiA8PSBJVEVSQVRJT05TX1BFUl9QUk9NUFQ7IGl0ZXJhdGlvbisrKSB7XHJcbiAgICAgIGNvbnNvbGUubG9nKGBbV29ya2VyICR7d29ya2VySWR9XSBSdW5uaW5nIGl0ZXJhdGlvbiAke2l0ZXJhdGlvbn0vJHtJVEVSQVRJT05TX1BFUl9QUk9NUFR9IGZvcjogXCIke3Byb21wdH1cImApO1xyXG5cclxuICAgICAgdHJ5IHtcclxuICAgICAgICAvLyBMYXVuY2ggbmV3IGJyb3dzZXIgY29udGV4dCBmb3IgZWFjaCBpdGVyYXRpb25cclxuICAgICAgICBjb25zdCBjb250ZXh0OiBCcm93c2VyQ29udGV4dCA9IGF3YWl0IGNocm9taXVtLmxhdW5jaFBlcnNpc3RlbnRDb250ZXh0KHVzZXJEYXRhRGlyLCB7XHJcbiAgICAgICAgICBjaGFubmVsOiAnY2hyb21lJyxcclxuICAgICAgICAgIGhlYWRsZXNzOiBmYWxzZSxcclxuICAgICAgICB9KTtcclxuXHJcbiAgICAgICAgY29uc3QgcGFnZSA9IGF3YWl0IGNvbnRleHQubmV3UGFnZSgpO1xyXG5cclxuICAgICAgICBhd2FpdCBwYWdlLmdvdG8od2Vic2l0ZSk7XHJcblxyXG4gICAgICAgIGF3YWl0IHBhZ2Uud2FpdEZvckxvYWRTdGF0ZSgnbG9hZCcpO1xyXG4gICAgICAgIGxldCBkb25lUmVsb2FkID0gZmFsc2VcclxuXHJcbiAgICAgICAgd2hpbGUgKCFkb25lUmVsb2FkKSB7XHJcbiAgICAgICAgICB0cnkge1xyXG4gICAgICAgICAgICBhd2FpdCBwYWdlLnJlbG9hZCgpO1xyXG4gICAgICAgICAgICBkb25lUmVsb2FkID0gdHJ1ZVxyXG4gICAgICAgICAgfSBjYXRjaCAoXykge1xyXG4gICAgICAgICAgICA7XHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgfVxyXG5cclxuICAgICAgICBhd2FpdCBwYWdlLmV2YWx1YXRlKCgpID0+IHtcclxuICAgICAgICAgIGRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlbW92ZScsIChlKSA9PiB7XHJcbiAgICAgICAgICAgIC8vIENyZWF0ZSBhIHJlZCBkb3QgZWxlbWVudFxyXG4gICAgICAgICAgICBjb25zb2xlLmxvZyhcImhpXCIpXHJcbiAgICAgICAgICAgIGNvbnN0IGRvdCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xyXG4gICAgICAgICAgICBkb3Quc3R5bGUucG9zaXRpb24gPSAnYWJzb2x1dGUnO1xyXG4gICAgICAgICAgICBkb3Quc3R5bGUud2lkdGggPSAnMTBweCc7XHJcbiAgICAgICAgICAgIGRvdC5zdHlsZS5oZWlnaHQgPSAnMTBweCc7XHJcbiAgICAgICAgICAgIGRvdC5zdHlsZS5iYWNrZ3JvdW5kQ29sb3IgPSAncmVkJztcclxuICAgICAgICAgICAgZG90LnN0eWxlLmJvcmRlclJhZGl1cyA9ICc1MCUnO1xyXG4gICAgICAgICAgICBkb3Quc3R5bGUubGVmdCA9IGAke2UuY2xpZW50WH1weGA7XHJcbiAgICAgICAgICAgIGRvdC5zdHlsZS50b3AgPSBgJHtlLmNsaWVudFl9cHhgO1xyXG4gICAgICAgICAgICBkb3Quc3R5bGUucG9pbnRlckV2ZW50cyA9ICdub25lJztcclxuICAgICAgICAgICAgZG90LnN0eWxlLnpJbmRleCA9ICc5OTk5JztcclxuXHJcbiAgICAgICAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQoZG90KTtcclxuXHJcbiAgICAgICAgICAgIC8vIE9wdGlvbmFsOiByZW1vdmUgdGhlIGRvdCBhZnRlciBhIHNob3J0IHRpbWVcclxuICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiBkb3QucmVtb3ZlKCksIDUwMCk7XHJcbiAgICAgICAgICB9KTtcclxuICAgICAgICB9KVxyXG4gICAgICAgIC8vIFJlcGxheSB1c2VyIGFjdGlvbnMgYW5kIGdldCByZXNwb25zZVxyXG4gICAgICAgIGNvbnN0IHJlc3BvbnNlID0gYXdhaXQgcmVwbGF5QWN0aW9uc0FuZEdldFJlc3BvbnNlKHBhZ2UsIGV2ZW50TG9ncywgcHJvbXB0KTtcclxuXHJcbiAgICAgICAgLy8gU3RvcmUgdGhlIHJlc3BvbnNlXHJcbiAgICAgICAgcmVzdWx0c1twcm9tcHRdLnB1c2gocmVzcG9uc2UpO1xyXG5cclxuICAgICAgICAvLyBDbG9zZSBicm93c2VyIGJlZm9yZSBuZXh0IGl0ZXJhdGlvblxyXG4gICAgICAgIGF3YWl0IGNvbnRleHQuY2xvc2UoKTtcclxuICAgICAgICBjb25zb2xlLmxvZyhgW1dvcmtlciAke3dvcmtlcklkfV0gQ29tcGxldGVkIGl0ZXJhdGlvbiAke2l0ZXJhdGlvbn0gZm9yOiBcIiR7cHJvbXB0fVwiYCk7XHJcblxyXG4gICAgICAgIC8vIFdhaXQgYmV0d2VlbiBpdGVyYXRpb25zIHRvIGF2b2lkIHJhdGUgbGltaXRpbmdcclxuICAgICAgICBhd2FpdCBuZXcgUHJvbWlzZShyZXNvbHZlID0+IHNldFRpbWVvdXQocmVzb2x2ZSwgMjAwMCkpO1xyXG5cclxuICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgICAgICBjb25zb2xlLmVycm9yKGBbV29ya2VyICR7d29ya2VySWR9XSBFcnJvciBpbiBpdGVyYXRpb24gJHtpdGVyYXRpb259IGZvciBcIiR7cHJvbXB0fVwiOmAsIChlcnJvciBhcyBFcnJvcikubWVzc2FnZSk7XHJcbiAgICAgICAgcmVzdWx0c1twcm9tcHRdLnB1c2goYEVycm9yIGluIGl0ZXJhdGlvbiAke2l0ZXJhdGlvbn06ICR7KGVycm9yIGFzIEVycm9yKS5tZXNzYWdlfWApO1xyXG4gICAgICB9XHJcbiAgICB9XHJcblxyXG4gICAgY29uc29sZS5sb2coYFtXb3JrZXIgJHt3b3JrZXJJZH1dIENvbXBsZXRlZCBhbGwgJHtJVEVSQVRJT05TX1BFUl9QUk9NUFR9IGl0ZXJhdGlvbnMgZm9yOiBcIiR7cHJvbXB0fVwiYCk7XHJcblxyXG4gICAgLy8gTG9uZ2VyIHdhaXQgYmV0d2VlbiBkaWZmZXJlbnQgcHJvbXB0c1xyXG4gICAgaWYgKHByb21wdHMuaW5kZXhPZihwcm9tcHQpIDwgcHJvbXB0cy5sZW5ndGggLSAxKSB7XHJcbiAgICAgIGF3YWl0IG5ldyBQcm9taXNlKHJlc29sdmUgPT4gc2V0VGltZW91dChyZXNvbHZlLCAzMDAwKSk7XHJcbiAgICB9XHJcbiAgfVxyXG5cclxuICByZXR1cm4gcmVzdWx0cztcclxufVxyXG5cclxuKGFzeW5jICgpID0+IHtcclxuICBjb25zb2xlLmxvZyhgW1dvcmtlciAke3dvcmtlcklkfV0gU3RhcnRpbmcgd2l0aCB1c2VyRGF0YURpcjogJHt1c2VyRGF0YURpcn1gKTtcclxuICBjb25zb2xlLmxvZyhgW1dvcmtlciAke3dvcmtlcklkfV0gQXNzaWduZWQgcHJvbXB0czpgLCBwcm9tcHRzKTtcclxuICBjb25zb2xlLmxvZyhgW1dvcmtlciAke3dvcmtlcklkfV0gV2lsbCBydW4gZWFjaCBwcm9tcHQgJHtJVEVSQVRJT05TX1BFUl9QUk9NUFR9IHRpbWVzYCk7XHJcblxyXG4gIC8vIENsZWFuIHVwIGFueSBleGlzdGluZyBicm93c2VyIGluc3RhbmNlc1xyXG4gIHRyeSB7XHJcbiAgICBjb25zdCBwcmVDb250ZXh0ID0gYXdhaXQgY2hyb21pdW0ubGF1bmNoUGVyc2lzdGVudENvbnRleHQodXNlckRhdGFEaXIsIHtcclxuICAgICAgY2hhbm5lbDogJ2Nocm9tZScsXHJcbiAgICAgIGhlYWRsZXNzOiB0cnVlLFxyXG4gICAgfSk7XHJcbiAgICBhd2FpdCBwcmVDb250ZXh0LmNsb3NlKCk7XHJcbiAgfSBjYXRjaCAoZXJyb3IpIHtcclxuICAgIGNvbnNvbGUubG9nKGBbV29ya2VyICR7d29ya2VySWR9XSBObyBleGlzdGluZyBjb250ZXh0IHRvIGNsZWFuIHVwYCk7XHJcbiAgfVxyXG5cclxuICBjb25zdCByZXN1bHRzID0gYXdhaXQgcnVuUHJvbXB0SXRlcmF0aW9ucygpO1xyXG5cclxuICAvLyBMb2cgc3VtbWFyeSBmb3IgdGhpcyB3b3JrZXJcclxuICBjb25zb2xlLmxvZyhgW1dvcmtlciAke3dvcmtlcklkfV0gQ09NUExFVEVEIEFMTCBXT1JLIWApO1xyXG4gIGNvbnNvbGUubG9nKGBbV29ya2VyICR7d29ya2VySWR9XSBTdW1tYXJ5OmApO1xyXG5cclxuICBPYmplY3Qua2V5cyhyZXN1bHRzKS5mb3JFYWNoKHByb21wdCA9PiB7XHJcbiAgICBjb25zb2xlLmxvZyhgICAtIFwiJHtwcm9tcHR9XCI6ICR7cmVzdWx0c1twcm9tcHRdLmxlbmd0aH0gcmVzcG9uc2VzIGNvbGxlY3RlZGApO1xyXG4gIH0pO1xyXG5cclxuICBwYXJlbnRQb3J0Py5wb3N0TWVzc2FnZShyZXN1bHRzKTtcclxufSkoKTsgIl0sIm5hbWVzIjpbInBhcmVudFBvcnQiLCJ3b3JrZXJEYXRhIiwiY2hyb21pdW0iLCJKU0RPTSIsImNyZWF0ZURPTVB1cmlmeSIsIndyaXRlRmlsZVN5bmMiLCJkb3RlbnYiLCJPcGVuQUkiLCJjb25maWciLCJldmVudExvZ3MiLCJ1c2VyRGF0YURpciIsIndvcmtlcklkIiwicHJvbXB0cyIsIndlYnNpdGUiLCJ3YWl0RHVyYXRpb24iLCJJVEVSQVRJT05TX1BFUl9QUk9NUFQiLCJyZXBsYXlBY3Rpb25zQW5kR2V0UmVzcG9uc2UiLCJwYWdlIiwibG9ncyIsInByb21wdFRleHQiLCJhY3Rpb25zVG9SZXBsYXkiLCJmaWx0ZXIiLCJldmVudCIsImluY2x1ZGVzIiwidHlwZSIsInNvcnQiLCJhIiwiYiIsInRpbWVzdGFtcCIsImNvbnNvbGUiLCJsb2ciLCJsZW5ndGgiLCJsYXN0VGltZXN0YW1wIiwiRGF0ZSIsIm5vdyIsImFjdGlvbiIsImRlbGF5IiwiUHJvbWlzZSIsInJlcyIsInNldFRpbWVvdXQiLCJNYXRoIiwibWluIiwieCIsInkiLCJkZXRhaWxzIiwibW91c2UiLCJtb3ZlIiwiZXZhbHVhdGUiLCJkb3duIiwiYnV0dG9uIiwiY2xpY2siLCJjbGlja09wdHMiLCJkYmxjbGljayIsImtleWJvYXJkIiwicHJlc3MiLCJrZXkiLCJuYXZpZ2F0b3IiLCJjbGlwYm9hcmQiLCJyZWFkVGV4dCIsImVyciIsImVycm9yIiwibWVzc2FnZSIsIndhaXRGb3JUaW1lb3V0IiwiY29udGVudCIsIndpbmRvdyIsIkRPTVB1cmlmeSIsImNsZWFuIiwic2FuaXRpemUiLCJBTExPV0VEX0FUVFIiLCJBTExPV19EQVRBX0FUVFIiLCJBTExPV19BUklBX0FUVFIiLCJzZWxlY3RlZFRleHQiLCJzZWxlY3Rpb24iLCJnZXRTZWxlY3Rpb24iLCJ0b1N0cmluZyIsInRyaW0iLCJzdWJzdHJpbmciLCJvcGVuYWkiLCJhcGlLZXkiLCJwcm9jZXNzIiwiZW52IiwiREVFUElORlJBX0FQSV9LRVkiLCJiYXNlVVJMIiwicmVwbHkiLCJjaGF0IiwiY29tcGxldGlvbnMiLCJjcmVhdGUiLCJtb2RlbCIsInRlbXBlcmF0dXJlIiwibWVzc2FnZXMiLCJyb2xlIiwic3RyZWFtIiwiY2hvaWNlcyIsInNwbGl0IiwiYXQiLCJydW5Qcm9tcHRJdGVyYXRpb25zIiwicmVzdWx0cyIsImZvckVhY2giLCJwcm9tcHQiLCJpdGVyYXRpb24iLCJjb250ZXh0IiwibGF1bmNoUGVyc2lzdGVudENvbnRleHQiLCJjaGFubmVsIiwiaGVhZGxlc3MiLCJuZXdQYWdlIiwiZ290byIsIndhaXRGb3JMb2FkU3RhdGUiLCJkb25lUmVsb2FkIiwicmVsb2FkIiwiXyIsImRvY3VtZW50IiwiYWRkRXZlbnRMaXN0ZW5lciIsImUiLCJkb3QiLCJjcmVhdGVFbGVtZW50Iiwic3R5bGUiLCJwb3NpdGlvbiIsIndpZHRoIiwiaGVpZ2h0IiwiYmFja2dyb3VuZENvbG9yIiwiYm9yZGVyUmFkaXVzIiwibGVmdCIsImNsaWVudFgiLCJ0b3AiLCJjbGllbnRZIiwicG9pbnRlckV2ZW50cyIsInpJbmRleCIsImJvZHkiLCJhcHBlbmRDaGlsZCIsInJlbW92ZSIsInJlc3BvbnNlIiwicHVzaCIsImNsb3NlIiwicmVzb2x2ZSIsImluZGV4T2YiLCJwcmVDb250ZXh0IiwiT2Jq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