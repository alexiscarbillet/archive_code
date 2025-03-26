// Task 1: Function-based console log message generator
function consoleStyler(color, background, fontSize, txt) {
    // 1. Create a message variable
    let message = "%c" + txt;

    // 2. Create a style variable
    let style = `color: ${color};`;

    // 3. Update the style with background
    style += `background: ${background};`;

    // 4. Update the style with font-size
    style += `font-size: ${fontSize};`;

    // 5. Log the message and style to the console
    console.log(message, style);
}

// Task 2: Another console log message generator
function celebrateStyler(reason) {
    // 1. Define the fontStyle variable
    let fontStyle = "color: tomato; font-size: 50px";

    // 2. Check if the reason is "birthday"
    if (reason === "birthday") {
        console.log(`%cHappy birthday`, fontStyle);
    }
    // 3. Check if the reason is "champions"
    else if (reason === "champions") {
        console.log(`%cCongrats on the title!`, fontStyle);
    }
    // 4. Default case
    else {
        let message = "%c" + reason;
        let style = `color: black; font-size: 20px;`;
        console.log(message, style);
    }
}

// Task 3: Run both the consoleStyler and celebrateStyler functions
consoleStyler('#1d5c63', '#ede6db', '40px', 'Congrats!');
celebrateStyler('birthday');

// Task 4: Insert a congratulatory and custom message
function styleAndCelebrate(color, background, fontSize, txt, reason) {
    // Invoke consoleStyler and celebrateStyler
    consoleStyler(color, background, fontSize, txt);
    celebrateStyler(reason);
}

// Invoke styleAndCelebrate with custom arguments
styleAndCelebrate('ef7c8e', 'fae8e0', '30px', 'You made it!', 'champions');
