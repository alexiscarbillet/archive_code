package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/chromedp/chromedp"
	"github.com/gocolly/colly/v2"
)

func main() {
	// URL of the Shopify store
	rootURL := "https://spirit-beauty-skincare.com/"

	// Initialize Colly Collector
	c := colly.NewCollector(
		colly.AllowedDomains("spirit-beauty-skincare.com"),
	)

	// Callback to handle the links on the homepage or collection pages
	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		link := e.Attr("href")
		// Check if the link leads to a product or collection page
		if isProductPage(link) {
			fmt.Println("Found product page:", link)
			e.Request.Visit(link)
		} else if isCollectionPage(link) {
			fmt.Println("Found collection page:", link)
			e.Request.Visit(link)
		}
	})

	// Callback for when a product page is visited
	c.OnHTML("html", func(e *colly.HTMLElement) {
		if isProductPage(e.Request.URL.String()) {
			fmt.Println("Visiting Product Page:", e.Request.URL.String())
			// Extract product details
			extractProductDetails(e)
		}
	})

	// Start by visiting the homepage
	err := c.Visit(rootURL)
	if err != nil {
		log.Fatal(err)
	}

	// Optionally, handle JS-heavy product pages (if necessary)
	automateCart()
}

// Check if the URL looks like a product page
func isProductPage(url string) bool {
	return strings.Contains(url, "/products/")
}

// Check if the URL is a collection page
func isCollectionPage(url string) bool {
	return strings.Contains(url, "/collections/")
}

// Extract product details from the product page
func extractProductDetails(e *colly.HTMLElement) {
	// Extract the title of the product
	title := e.ChildText("h1.product-title")
	if title != "" {
		fmt.Println("Product Title:", title)
	} else {
		fmt.Println("Product Title: Not found")
	}

	// Extract the price of the product
	price := e.ChildText("span.price")
	if price != "" {
		fmt.Println("Product Price:", price)
	} else {
		fmt.Println("Product Price: Not found")
	}

	// Extract the product description
	description := e.ChildText("div.product-description")
	if description != "" {
		fmt.Println("Product Description:", description)
	} else {
		fmt.Println("Product Description: Not found")
	}
}

// Use chromedp to automate JS-heavy pages (if needed)
func automateCart() {
	// Create a new context
	ctx, cancel := chromedp.NewContext(context.Background())
	defer cancel()

	// Run the automation (optional if there's JS content)
	err := chromedp.Run(ctx,
		chromedp.Navigate("https://spirit-beauty-skincare.com/products/some-product"), // Replace with actual product page
		chromedp.Click("#add-to-cart-button"),                                         // Click "Add to Cart" button
		chromedp.Sleep(2),                                                             // Sleep to allow the action to complete
	)
	if err != nil {
		log.Fatal("Error automating cart:", err)
	}
	fmt.Println("Product added to cart via chromedp.")
}
