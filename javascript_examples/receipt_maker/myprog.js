// Given variables
const dishData = [
    {
        name: "Italian pasta",
        price: 9.55
    },
    {
        name: "Rice with veggies",
        price: 8.65
    },
    {
        name: "Chicken with potatoes",
        price: 15.55
    },
    {
        name: "Vegetarian Pizza",
        price: 6.45
    },
]
const tax = 1.20;

// Implement getPrices()
// Step 2: Function to get prices, applying tax if needed
function getPrices(taxBoolean) {

  // Step 3: Loop through all dishes
  for (let dish of dishData) {
    let finalPrice;

    // Step 4: If taxBoolean is true, apply tax
    if (taxBoolean === true) {
      finalPrice = dish.price * (1 + tax); // Adding tax to price
    }
    // Step 5: If taxBoolean is false, no tax, just the price
    else if (taxBoolean === false) {
      finalPrice = dish.price;
    }
    // Step 6: Handle invalid taxBoolean value
    else {
      console.log("You need to pass a boolean to the getPrices call!");
      return; // Exit function early if taxBoolean is invalid
    }

    // Step 7: Log the dish name and final price
    console.log(`Dish: ${dish.name} Price: $${finalPrice.toFixed(2)}`);
  }
}


// Implement getDiscount()
function getDiscount(taxBoolean, guests) {
  // Step 9: Call getPrices() inside getDiscount()
  getPrices(taxBoolean);

  // Step 10: Check if guests is a number and between 1 and 30
  if (typeof guests === 'number' && guests > 0 && guests < 30) {
    let discount = 0;

    // Step 11: Calculate discount based on the number of guests
    if (guests < 5) {
      discount = 5;
    } else if (guests >= 5) {
      discount = 10;
    }

    // Log the discount
    console.log(`Discount is: $${discount}`);
  }
  // Step 12: Handle invalid guests parameter
  else {
    console.log('The second argument must be a number between 0 and 30');
  }
}
// Call getDiscount()
// Case 1: Tax is applied, guests are 2
getDiscount(true, 2);

// Case 2: No tax applied, guests are 10
getDiscount(false, 10);

// Case 3: Invalid tax argument, should print error
getDiscount('yes', 10);

// Case 4: Invalid guests number (greater than 30), should print error
getDiscount(true, 31);

// Case 5: No guests provided, should print error
getDiscount(true);