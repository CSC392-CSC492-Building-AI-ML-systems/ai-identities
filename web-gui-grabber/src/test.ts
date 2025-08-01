import { chromium, Page } from 'patchright';
import path from 'path';
import fs from 'fs';
import { Worker } from 'worker_threads';

declare global {
  interface Window {
    replayMode: boolean;
    logEvent: (type: EventType, details?: EventDetails) => void;
    closeBrowser: () => void;
    
  }
}

const args = process.argv.slice(2); // skip the first two default entries

const website = args[0]
const num_workers = parseInt(args[1]);
const base_profile = args[2]
const profiles_clones_dir = args[3]
const eventLogs: CustomEventLog[] = [];



// 10 prompts total - 2 per worker

const promptArrTxt = fs.readFileSync(args[4], 'utf-8');
const ALL_PROMPTS: string[] = JSON.parse(promptArrTxt)

console.log(args)
const waitDuration: number = parseInt(args[5])
console.log(waitDuration)

let hasRegistered = false;

async function createProfileCopies(): Promise<string[]> {
  
  if (!fs.existsSync(profiles_clones_dir)) {
    fs.mkdirSync(profiles_clones_dir, { recursive: true });
  }

  const dirs: string[] = [];

  for (let i = 0; i < num_workers; i++) {
    const cloneDir = path.join(profiles_clones_dir, `profile-${i}`);

    dirs.push(cloneDir);
  }

  await Promise.all(
    dirs.map((cloneDir) =>
      fs.rm(cloneDir, ()=>{})
    )
  );

  await Promise.all(
    dirs.map((cloneDir) =>
      fs.cp(base_profile, cloneDir, ()=>{})
    )
  );

  return dirs;
}

function spawnWorkers(profileDirs: string[]) {
  let promptList: string[][] = []

  for (const _ of profileDirs) {
    promptList.push([])
  }

  for (let i = 0; i < ALL_PROMPTS.length; i++) {
    promptList[i % profileDirs.length].push(ALL_PROMPTS[i])
  }

  const promises = profileDirs.map((profileDir, i) => {
    return new Promise<Record<string, string[]>>((resolve, reject) => {
      let workerPrompts = promptList[i]
      const worker = new Worker('./build/src/worker.js', {
        workerData: {
          eventLogs,
          userDataDir: profileDir,
          workerId: i,
          prompts: promptList[i],
          website: website,
          waitDuration: waitDuration 
        },
      });

      console.log(`Worker ${i} started with profile ${profileDir} and prompts:`, workerPrompts);

      worker.on('message', (msg) => {
        console.log(`Worker ${i} finished with results for ${Object.keys(msg).length} prompts`);
        resolve(msg);
      });

      worker.on('error', reject);
      worker.on('exit', (code) => {
        if (code !== 0) reject(new Error(`Worker ${i} exited with code ${code}`));
      });
    });
  });

  Promise.all(promises)
    .then((results) => {
      // Combine all worker results into a single JSON object
      // Structure: { "prompt text": ["response1", "response2", "response3", "response4"] }
      const combinedResults: Record<string, string[]> = {};

      results.forEach((workerResult) => {
        Object.keys(workerResult).forEach(prompt => {
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
      Object.keys(combinedResults).forEach(prompt => {
        console.log(`\nPrompt: "${prompt}"`);
        console.log(`Responses collected: ${combinedResults[prompt].length}`);
        combinedResults[prompt].forEach((response, index) => {
          console.log(`  ${index + 1}: ${response.substring(0, 100)}${response.length > 100 ? '' : ''}`);
        });
      });
    })
    .catch((error) => {
      console.error('Error in workers:', error);
    });
}

(async () => {

  const context = await chromium.launchPersistentContext(base_profile, {
    channel: 'chrome',
    headless: false,
    viewport:{height:0, width:0}
  });

  async function setupEventListeners(page: Page) {
    if (!hasRegistered) {
      await page.exposeFunction('logEvent', async (type: EventType, details?: EventDetails) => {

        if (!details) {
          details = { message: type }
        }

        let logEntry: CustomEventLog = { type, details, timestamp: Date.now() };

        eventLogs.push(logEntry);
      });

      await page.exposeFunction('closeBrowser', () => {
        context.close().then(async () => {

          const profileDirs = createProfileCopies();
          let worker_dirs = []
          for (let i = 0; i < num_workers; i++) {
            worker_dirs.push('./cloned-profiles/profile-' + i.toString())
          }
          spawnWorkers(await profileDirs);
        })
      })

      hasRegistered = true;
    }

    await page.evaluate(() => {
      console.log("Attaching listeners...");

      // Remove old listeners if they exist
      if ((window as any)._mouseMoveListener) {
        document.removeEventListener('mousemove', (window as any)._mouseMoveListener);
        document.removeEventListener('click', (window as any)._clickListener);
        document.removeEventListener('dblclick', (window as any)._dblClickListener);
        document.removeEventListener('keydown', (window as any)._keyDownListener);
      }

      // Define listeners
      const mouseMoveListener = (e: MouseEvent) => {
        try {
          window.logEvent('mousemove', { x: e.clientX, y: e.clientY });
        } catch (e) {
          console.error(e)
          console.log("FAILED")
          window.location.reload()
        }
      };

      const clickListener = (e: MouseEvent) => {
        try {
          window.logEvent('click', { x: e.clientX, y: e.clientY, button: e.button });
        } catch (e) {
          console.error(e)
          console.log("FAILED")
          window.location.reload()
        }
      }


      const dblClickListener = (e: MouseEvent) => {
        try {
          window.logEvent('dblclick', { x: e.clientX, y: e.clientY, button: e.button });
        } catch (e) {
          console.error(e)
          console.log("FAILED")
          window.location.reload()
        }

      };

      const copyListener = async (e: ClipboardEvent) => {
        window.logEvent('copy')
        console.log(await navigator.clipboard.readText())
      }
      const keyDownListener = (e: KeyboardEvent) => {
        if (e.key === 'F12') {
          e.preventDefault();
          console.log('F12 pressed - entering replay mode');
          window.replayMode = true;
          window.logEvent('f12_pressed', {
            message: 'Entered replay mode - press any key to start replay',
          });
        } else if (e.key === 'Escape') {
          window.logEvent('escape_pressed', { message: 'Closing browser' });
        } else if (e.key === 'Enter') {
          window.logEvent('enter_key_press', { key: e.key });
        } else if (window.replayMode) {
          e.preventDefault();
          window.logEvent('replay_trigger', {
            key: e.key,
            message: 'Starting replay...',
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
      (window as any)._mouseMoveListener = mouseMoveListener;
      (window as any)._clickListener = clickListener;
      (window as any)._dblClickListener = dblClickListener;
      (window as any)._keyDownListener = keyDownListener;

      // Attach new listeners
      document.addEventListener('mousemove', mouseMoveListener);
      document.addEventListener('click', clickListener);
      document.addEventListener('dblclick', dblClickListener);
      document.addEventListener('keydown', keyDownListener);
      document.addEventListener('copy', copyListener)
    });
  }

  const page = await context.newPage();

  page.on('framenavigated', async () => {
    try {
      await setupEventListeners(page);
    }
    catch (_) {
      ;
    }

  });

  await page.goto(website);

  await page.waitForLoadState('load');
  let doneReload = false

  console.log('Loading script...')
  while(!doneReload){
    try{
        await page.reload();
        doneReload = true
    } catch(_){
      ;
    }
  }

  await page.setViewportSize({ width: 1280, height: 720 });
  console.log('Browser opened. Interact with the page:');
  console.log('- Record your interactions normally');
  console.log('- Press F12 to enter replay mode');
  console.log('- Press any key after F12 to start workers');
  console.log(`- Each worker will run 2 prompts, 4 times each`);
  console.log(`- Total: ${num_workers} workers × 2 prompts × 4 iterations = ${num_workers * 2 * 4} responses`);
})()