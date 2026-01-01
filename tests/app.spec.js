const { test, expect } = require('@playwright/test');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const PORT = 8765;
const BASE_URL = `http://localhost:${PORT}`;
let serverProcess;

async function waitForServer(url, attempts = 20) {
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

test.beforeAll(async () => {
  serverProcess = spawn('python', ['app.py'], {
    env: { ...process.env, PORT: PORT.toString() },
    stdio: 'inherit',
  });
  await waitForServer(BASE_URL);
});

test.afterAll(() => {
  if (serverProcess) {
    serverProcess.kill('SIGTERM');
  }
});

test('uploads an image, records a selection, and downloads predictions', async ({ page }) => {
  const tmpDir = path.join(__dirname, 'tmp');
  fs.mkdirSync(tmpDir, { recursive: true });
  const imgPath = path.join(tmpDir, 'tiny.png');
  const redDotPng =
    'iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAI0lEQVQoU2P8z8AARMiABRhGoxGkAkmkSCaRaIJEJoFoAAAjVgw/Y+ny8gAAAABJRU5ErkJggg==';
  fs.writeFileSync(imgPath, Buffer.from(redDotPng, 'base64'));

  await page.goto(BASE_URL);
  await expect(page.getByText('SAM 3 Image Segmentor')).toBeVisible();

  const fileInput = page.locator('input[type="file"][name="image"]');
  await fileInput.setInputFiles(imgPath);
  await page.getByRole('button', { name: 'Upload' }).click();
  await page.waitForURL(`${BASE_URL}/`);

  const canvas = page.locator('canvas#overlay');
  await canvas.waitFor({ state: 'visible' });
  const box = await canvas.boundingBox();
  await canvas.click({ position: { x: box.width / 2, y: box.height / 2 } });
  await expect(page.locator('#selections')).toHaveValue(/positive/);

  await page.getByRole('button', { name: 'Run segmentation' }).click();
  await expect(page.locator('#predictions')).not.toHaveValue('');

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download predictions' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('predictions.json');
});
