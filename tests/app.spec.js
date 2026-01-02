const { test, expect } = require('@playwright/test');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const REPO_ROOT = path.resolve(__dirname, '..');
const TMP_DIR = path.join(__dirname, 'tmp');
const PORT = Number(process.env.E2E_PORT || 8765);
const BASE_URL = process.env.E2E_BASE_URL || `http://localhost:${PORT}`;
const SEGMENT_TIMEOUT_MS = Number(process.env.E2E_SEGMENT_TIMEOUT_MS || 180000);
const TEST_TIMEOUT_MS = Number(process.env.E2E_TEST_TIMEOUT_MS || SEGMENT_TIMEOUT_MS + 60000);
const SAMPLE_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAAAoUlEQVR42u3RMREAAAgEoO9fWiM4OehBBRIAAAAAAAAAAAAAAIBBHSdAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgQIECBAgAABAgR8CAAAAAAAAAAAAAAAANY1dZ7Swq6HsBgAAAAASUVORK5CYII=';

let serverProcess;

function readSecret(name) {
  const secretPath = path.join(REPO_ROOT, '.secrets', name);
  if (!fs.existsSync(secretPath)) return null;
  return fs.readFileSync(secretPath, 'utf8').trim();
}

function buildAppEnv() {
  const env = { ...process.env, PORT: PORT.toString() };
  const modalTokenId = env.MODAL_TOKEN_ID || readSecret('modal_token_id');
  const modalTokenSecret = env.MODAL_TOKEN_SECRET || readSecret('modal_token_secret');
  const hfToken = env.HUGGINGFACE_TOKEN || env.HF_TOKEN || readSecret('huggingface_token');
  if (modalTokenId) env.MODAL_TOKEN_ID = modalTokenId;
  if (modalTokenSecret) env.MODAL_TOKEN_SECRET = modalTokenSecret;
  if (hfToken) {
    env.HUGGINGFACE_TOKEN = hfToken;
    if (!env.HF_TOKEN) env.HF_TOKEN = hfToken;
  }
  if (!env.MODAL_LOGLEVEL) env.MODAL_LOGLEVEL = 'DEBUG';
  if (!env.MODAL_TRACEBACK) env.MODAL_TRACEBACK = '1';
  return env;
}

const SECRETS_DIR = path.join(REPO_ROOT, '.secrets');
const SECRETS_DIR_PRESENT = fs.existsSync(SECRETS_DIR);
const APP_ENV = buildAppEnv();
const MODAL_TOKEN_ID_PRESENT = Boolean(APP_ENV.MODAL_TOKEN_ID);
const MODAL_TOKEN_SECRET_PRESENT = Boolean(APP_ENV.MODAL_TOKEN_SECRET);
const MODAL_AVAILABLE = Boolean(MODAL_TOKEN_ID_PRESENT && MODAL_TOKEN_SECRET_PRESENT);
const SKIP_MODAL = process.env.E2E_SKIP_MODAL === '1';

console.log(
  `[e2e] secrets dir=${SECRETS_DIR_PRESENT ? 'found' : 'missing'}(${SECRETS_DIR}) ` +
    `modal_token_id=${MODAL_TOKEN_ID_PRESENT ? 'set' : 'missing'} ` +
    `modal_token_secret=${MODAL_TOKEN_SECRET_PRESENT ? 'set' : 'missing'} ` +
    `huggingface_token=${APP_ENV.HUGGINGFACE_TOKEN ? 'set' : 'missing'}`
);

async function waitForServer(url, attempts = 40) {
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch (err) {
      // ignore until reachable
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error('Server did not start in time');
}

test.describe('segmentation app e2e', () => {
  test.skip(SKIP_MODAL, 'Skipping Modal-backed E2E tests because E2E_SKIP_MODAL=1.');

  test.beforeAll(async () => {
    if (!MODAL_AVAILABLE) {
      throw new Error(
        'Modal credentials missing. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, or use E2E_SKIP_MODAL=1 to skip.'
      );
    }

    fs.mkdirSync(TMP_DIR, { recursive: true });
    const pythonBin = process.env.PYTHON || 'python';
    serverProcess = spawn(pythonBin, ['app.py'], {
      cwd: REPO_ROOT,
      env: APP_ENV,
      stdio: 'inherit',
    });
    await waitForServer(BASE_URL);
  });

  test.afterAll(() => {
    if (serverProcess) {
      serverProcess.kill('SIGTERM');
    }
  });

  test('uploads, captures selections, segments via Modal, and downloads predictions', async ({ page }) => {
    test.setTimeout(TEST_TIMEOUT_MS);

    const imgPath = path.join(TMP_DIR, 'sample.png');
    fs.writeFileSync(imgPath, Buffer.from(SAMPLE_PNG_BASE64, 'base64'));

    await page.goto(BASE_URL);
    await expect(page.getByText('SAM 3 Image Segmentor')).toBeVisible();

    await page.locator('input[type="file"][name="image"]').setInputFiles(imgPath);
    await page.getByRole('button', { name: 'Upload' }).click();
    await page.waitForURL(`${BASE_URL}/`);

    const canvas = page.locator('canvas#canvas');
    await canvas.waitFor({ state: 'visible' });
    const bounds = await canvas.boundingBox();
    if (!bounds) {
      throw new Error('Canvas not visible after upload.');
    }

    // Add text prompt
    await page.fill('#text-prompt', 'white square');

    // Add a point (default mode is now box, so switch to point mode first)
    await page.selectOption('#mode-select', 'point');
    await canvas.click({ position: { x: bounds.width * 0.5, y: bounds.height * 0.5 } });

    // Add a box with negative polarity
    await page.selectOption('#mode-select', 'box');
    await page.selectOption('#polarity-select', 'negative');
    // For box drawing: mousedown at start, then mouseup at end
    await canvas.hover({ position: { x: bounds.width * 0.25, y: bounds.height * 0.25 } });
    await page.mouse.down();
    await canvas.hover({ position: { x: bounds.width * 0.75, y: bounds.height * 0.75 } });
    await page.mouse.up();

    await page.getByRole('button', { name: 'Run' }).click();
    await expect
      .poll(async () => {
        const text = await page.locator('#predictions').inputValue();
        if (!text) return 0;
        try {
          return JSON.parse(text).width || 0;
        } catch (err) {
          return 0;
        }
      }, { timeout: SEGMENT_TIMEOUT_MS })
      .toBeGreaterThan(0);

    const predictions = JSON.parse(await page.locator('#predictions').inputValue());
    expect(predictions.width).toBe(128);
    expect(predictions.height).toBe(128);
    // The modal returns prompt_points and prompt_boxes as arrays
    expect(Array.isArray(predictions.prompt_points)).toBe(true);
    expect(Array.isArray(predictions.prompt_boxes)).toBe(true);
    // We added 1 point and 1 box
    expect(predictions.prompt_points.length).toBeGreaterThanOrEqual(1);
    expect(predictions.prompt_boxes.length).toBeGreaterThanOrEqual(1);
    // First point should be positive
    if (predictions.prompt_points.length > 0) {
      expect(predictions.prompt_points[0][2]).toBe('positive');
    }
    // First box should be negative
    if (predictions.prompt_boxes.length > 0) {
      expect(predictions.prompt_boxes[0][4]).toBe('negative');
    }
    expect(predictions.text_prompts).toContain('white square');
    expect(Array.isArray(predictions.masks)).toBe(true);

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download JSON' }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('predictions.json');
    const downloadPath = await download.path();
    expect(downloadPath).not.toBeNull();
    const downloadedPayload = fs.readFileSync(downloadPath, 'utf8');
    // Compare parsed JSON to avoid formatting differences
    const downloadedJson = JSON.parse(downloadedPayload);
    const predictionsJson = JSON.parse(await page.locator('#predictions').inputValue());
    expect(downloadedJson).toEqual(predictionsJson);
  });
});
