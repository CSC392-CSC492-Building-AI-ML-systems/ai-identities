import { parentPort, workerData } from 'worker_threads';
import { chromium, Page, BrowserContext } from 'patchright';
// sanitize.js
import { JSDOM } from 'jsdom';
import createDOMPurify from 'dompurify';
import { writeFileSync } from 'fs'; // named import
import dotenv from "dotenv";
import OpenAI from 'openai';
dotenv.config();  // load .env into process.env
// Create a virtual DOM window


const eventLogs: CustomEventLog[] = workerData.eventLogs;
const userDataDir: string = workerData.userDataDir;
const workerId: number = workerData.workerId;
const prompts: string[] = workerData.prompts;
const website: string = workerData.website
const waitDuration: number = workerData.waitDuration
const ITERATIONS_PER_PROMPT = 1;

async function replayActionsAndGetResponse(page: Page, logs: CustomEventLog[], promptText: string): Promise<string> {
  try {
    const actionsToReplay = logs
      .filter((event) => !['f12_pressed', 'escape_pressed', 'replay_trigger'].includes(event.type))
      .sort((a, b) => a.timestamp - b.timestamp);

    console.log(`[Worker ${workerId}] Replaying ${actionsToReplay.length} actions for prompt: "${promptText}"`);

    let lastTimestamp = actionsToReplay[0]?.timestamp ?? Date.now();

    for (const action of actionsToReplay) {
      try {
        const delay = action.timestamp - lastTimestamp;
        if (delay > 0) {
          await new Promise((res) => setTimeout(res, Math.min(delay, 1000))); // Cap delay at 1 second
        }
        lastTimestamp = action.timestamp;

        switch (action.type) {
          case 'mousemove': {
            const { x, y } = action.details as { x: number; y: number };
            await page.mouse.move(x, y);
            console.log(await page.evaluate("window.getSelection().toString();"))
            break;
          }
          case 'mousedown': {
            await page.mouse.down()

          }
          case 'click': {
            const { x, y, button } = action.details as { x: number; y: number; button: number };
            await page.mouse.click(x, y, {
              button: button === 0 ? 'left' : button === 1 ? 'middle' : 'right',
            });
            break;
          }
          case 'dblclick': {
            const { x, y, button } = action.details as { x: number; y: number; button: number };
            const clickOpts: { button?: "left" | "middle" | "right" | undefined } = {
              button: button === 0 ? 'left' : button === 1 ? 'middle' : 'right',
            };
            await page.mouse.dblclick(x, y, clickOpts);
            break;
          }
          case 'enter_key_press': {
            // Type our custom prompt text instead of default text
            await page.keyboard.type(promptText);
            await page.keyboard.press('Enter');
            break;
          }
          case 'keydown': {
            const { key } = action.details as { key: string };
            await page.keyboard.press(key);
            break;
          }

          case 'copy': {
            console.log(await page.evaluate("window.getSelection().toString();"))
            await page.keyboard.press("Control+C")
            console.log("hello")
            await page.evaluate(async () => {
              console.log(await navigator.clipboard.readText())
            })
            console.log("hello")
          }

        }
      } catch (err) {
        console.error(`[Worker ${workerId}] Error during ${action.type}:`, (err as Error).message);
      }
    }

    // Wait for ChatGPT to respond
    console.log(`[Worker ${workerId}] Waiting for response to: "${promptText}"`);

    await page.waitForTimeout(waitDuration); // Wait longer for AI response
    
    let content = await page.content()
    const window = new JSDOM('').window;
    const DOMPurify = createDOMPurify(window);

    const clean = DOMPurify.sanitize(content, { ALLOWED_ATTR: [], ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false });

    writeFileSync(Date.now() + ".html", clean)
    // Get selected text using the same method as the original code
    const selectedText = await page.evaluate(() => {
      const selection = window.getSelection();
      return selection ? selection.toString().trim() : '';
    });

    console.log(`[Worker ${workerId}] Selected text after replay: "${selectedText.substring(0, 100)}"`);

    const openai = new OpenAI({
      apiKey: process.env.DEEPINFRA_API_KEY,
      baseURL: "https://api.deepinfra.com/v1/openai",
    });

    const reply = await openai.chat.completions.create({
      model: "Qwen/Qwen3-235B-A22B",
      temperature:0.0,
      messages: [
        { role: "user", content: `Extract the response of the LLM from the following conversation, which is formatted in HTML. Answer with ONLY the response of the LLM, nothing else. In your answer, make sure to include "prefix phrases" such as "sure, let me help you with that", and "suffix phrases" such as "can I help you with anything else" if the LLM in the conversation responded with those phrases. Include LLM response text in your answer only if it is meant to the visible to the end user. Here is the conversation: ${clean}` }
      ],
      stream: false,
    });


    return reply.choices[0].message.content?.split("</think>").at(-1) as string

  } catch (error) {
    console.error(`[Worker ${workerId}] Error replaying actions for "${promptText}":`, (error as Error).message);
    return `Error: ${(error as Error).message}`;
  }
}

async function runPromptIterations(): Promise<Record<string, string[]>> {
  const results: Record<string, string[]> = {};

  // Initialize results structure for both prompts
  prompts.forEach(prompt => {
    results[prompt] = [];
  });

  // Run each prompt 4 times
  for (const prompt of prompts) {
    console.log(`[Worker ${workerId}] Starting iterations for prompt: "${prompt}"`);

    for (let iteration = 1; iteration <= ITERATIONS_PER_PROMPT; iteration++) {
      console.log(`[Worker ${workerId}] Running iteration ${iteration}/${ITERATIONS_PER_PROMPT} for: "${prompt}"`);

      try {
        // Launch new browser context for each iteration
        const context: BrowserContext = await chromium.launchPersistentContext(userDataDir, {
          channel: 'chrome',
          headless: false,
        });

        const page = await context.newPage();

        await page.goto(website);

        await page.waitForLoadState('load');
        let doneReload = false

        while (!doneReload) {
          try {
            await page.reload();
            doneReload = true
          } catch (_) {
            ;
          }
        }

        await page.evaluate(() => {
          document.addEventListener('mousemove', (e) => {
            // Create a red dot element
            console.log("hi")
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
            setTimeout(() => dot.remove(), 500);
          });
        })
        // Replay user actions and get response
        const response = await replayActionsAndGetResponse(page, eventLogs, prompt);

        // Store the response
        results[prompt].push(response);

        // Close browser before next iteration
        await context.close();
        console.log(`[Worker ${workerId}] Completed iteration ${iteration} for: "${prompt}"`);

        // Wait between iterations to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 2000));

      } catch (error) {
        console.error(`[Worker ${workerId}] Error in iteration ${iteration} for "${prompt}":`, (error as Error).message);
        results[prompt].push(`Error in iteration ${iteration}: ${(error as Error).message}`);
      }
    }

    console.log(`[Worker ${workerId}] Completed all ${ITERATIONS_PER_PROMPT} iterations for: "${prompt}"`);

    // Longer wait between different prompts
    if (prompts.indexOf(prompt) < prompts.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
  }

  return results;
}

(async () => {
  console.log(`[Worker ${workerId}] Starting with userDataDir: ${userDataDir}`);
  console.log(`[Worker ${workerId}] Assigned prompts:`, prompts);
  console.log(`[Worker ${workerId}] Will run each prompt ${ITERATIONS_PER_PROMPT} times`);

  // Clean up any existing browser instances
  try {
    const preContext = await chromium.launchPersistentContext(userDataDir, {
      channel: 'chrome',
      headless: true,
    });
    await preContext.close();
  } catch (error) {
    console.log(`[Worker ${workerId}] No existing context to clean up`);
  }

  const results = await runPromptIterations();

  // Log summary for this worker
  console.log(`[Worker ${workerId}] COMPLETED ALL WORK!`);
  console.log(`[Worker ${workerId}] Summary:`);

  Object.keys(results).forEach(prompt => {
    console.log(`  - "${prompt}": ${results[prompt].length} responses collected`);
  });

  parentPort?.postMessage(results);
})(); 