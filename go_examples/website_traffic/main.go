package main

import (
	"fmt"
	"net/http"
	"sync"
)

func sendRequest(url string, wg *sync.WaitGroup) {
	defer wg.Done()
	resp, err := http.Get(url)
	if err != nil {
		fmt.Printf("Error sending request to %s: %v\n", url, err)
		return
	}
	defer resp.Body.Close()
	fmt.Printf("Request sent to %s, Status: %s\n", url, resp.Status)
}

func main() {
	// Define the URL of the website to generate traffic
	websiteURL := "https://ac-electricity.com/"

	// Define the number of concurrent requests to send
	numRequests := 100

	// Create a wait group to wait for all goroutines to finish
	var wg sync.WaitGroup
	wg.Add(numRequests)

	// Send multiple concurrent requests to the website
	for i := 0; i < numRequests; i++ {
		go sendRequest(websiteURL, &wg)
	}

	// Wait for all goroutines to finish
	wg.Wait()

	websiteURL2 := "https://alexis-carbillet.com/"

	// Define the number of concurrent requests to send
	numRequests2 := 100

	// Create a wait group to wait for all goroutines to finish
	var wg2 sync.WaitGroup
	wg.Add(numRequests2)

	// Send multiple concurrent requests to the website
	for i := 0; i < numRequests2; i++ {
		go sendRequest(websiteURL2, &wg2)
	}

	// Wait for all goroutines to finish
	wg2.Wait()
}
