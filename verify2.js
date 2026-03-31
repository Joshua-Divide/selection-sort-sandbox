import puppeteer from 'puppeteer';

(async () => {
    try {
        const browser = await puppeteer.launch({ headless: "new" });
        const page = await browser.newPage();

        await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
        await page.waitForSelector('.visualization'); // random class guess
        
        const bodyHTML = await page.evaluate(() => document.body.innerText);
        console.log("Extracted page text:\n", bodyHTML.substring(0, 500));
        
        await browser.close();
        process.exit(0);
    } catch (e) {
        console.log("Trying generic wait");
        try {
            const browser = await puppeteer.launch({ headless: "new" });
            const page = await browser.newPage();
            await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
            await new Promise(r => setTimeout(r, 2000));
            const bodyHTML = await page.evaluate(() => document.body.innerText);
            console.log("Extracted page text:\n", bodyHTML.substring(0, 500));
            await browser.close();
        } catch(err) {
            console.error("Puppeteer Script Error:", err);
            process.exit(1);
        }
    }
})();