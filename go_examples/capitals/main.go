package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Define a map of countries and their capitals
	capitals := map[string]string{
		"France":    "Paris",
		"Germany":   "Berlin",
		"Italy":     "Rome",
		"Japan":     "Tokyo",
		"UK":        "London",
		"USA":       "Washington D.C.",
		"Canada":    "Ottawa",
		"Australia": "Canberra",
		"India":     "New Delhi",
		"China":     "Beijing",
	}

	// Create a slice to store country names
	countries := make([]string, 0, len(capitals))
	for country := range capitals {
		countries = append(countries, country)
	}

	// Randomly select a country
	randomIndex := rand.Intn(len(countries))
	randomCountry := countries[randomIndex]

	// Get the capital of the selected country
	correctCapital := capitals[randomCountry]

	// Ask the user to input the capital
	fmt.Printf("What is the capital of %s?\n", randomCountry)
	var userCapital string
	fmt.Scanln(&userCapital)

	// Check if the user's input matches the correct capital
	if userCapital == correctCapital {
		fmt.Println("Correct!")
	} else {
		fmt.Printf("Incorrect. The correct capital is %s.\n", correctCapital)
	}
}
