var dairy = ['cheese', 'sour cream', 'milk', 'yogurt', 'ice cream', 'milkshake'];

function logDairy() {
  for (const item of dairy) {
    console.log(item);
  }
}

logDairy();

const animal = {
    canJump: true
};
  
const bird = Object.create(animal);
bird.canFly = true;
bird.hasFeathers = true;

function birdCan() {
    for (const property of Object.keys(bird)) {
        console.log(`${property}: ${bird[property]}`);
    }
}

birdCan();

  
function animalCan() {
    for (const property in bird) {
        console.log(`${property}: ${bird[property]}`);
    }
}

animalCan();

  
