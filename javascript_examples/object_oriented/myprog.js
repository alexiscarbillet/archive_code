// Task 1: Code a Person class
class Person {
    constructor(name = "Tom", age = 20, energy = 100) {
        this.name = name;
        this.age = age;
        this.energy = energy;
    }

    // Method to increase energy
    sleep() {
        this.energy += 10;
        console.log(`${this.name} slept. Energy is now ${this.energy}`);
    }

    // Method to decrease energy
    doSomethingFun() {
        this.energy -= 10;
        console.log(`${this.name} did something fun. Energy is now ${this.energy}`);
    }
}

// Task 2: Code a Worker class that inherits from Person
class Worker extends Person {
    constructor(name, age, energy, xp = 0, hourlyWage = 10) {
        super(name, age, energy);
        this.xp = xp;
        this.hourlyWage = hourlyWage;
    }

    // Method to increase experience points when the worker goes to work
    goToWork() {
        this.xp += 10;
        console.log(`${this.name} went to work. Experience is now ${this.xp}`);
    }
}

// Task 3: Code an intern object
function intern() {
    const internObject = new Worker("Bob", 21, 110, 0, 10);
    internObject.goToWork(); // Run goToWork() method
    return internObject;
}

// Task 4: Code a manager object
function manager() {
    const managerObject = new Worker("Alice", 30, 120, 100, 30);
    managerObject.doSomethingFun(); // Run doSomethingFun() method
    return managerObject;
}

// Example usage:
const intern1 = intern();
console.log(intern1);

const manager1 = manager();
console.log(manager1);
