package main

import (
	"fmt"
	"strings"
)

func main() {
	string1 := "az*e342ty$340$vf$yo$djsql$23322$09*8"
	results := map[string]int{
		"n": 0,
		"l": 0,
		"o": 0,
	}
	shouldCount := true
	for _, i := range string1 {
		char := string(i)

		if char == "$" {
			shouldCount = !shouldCount
		} else {
			if shouldCount {
				if strings.Contains("1234567890", char) {
					results["n"]++
				} else if strings.Contains("azertyuiopqsdfghjklmwxcvbn", char) {
					results["l"]++
				} else {
					results["o"]++
				}
			} else {

			}
		}
	}
	fmt.Println(results) // should get [l:12 n:6 o:2]
}
