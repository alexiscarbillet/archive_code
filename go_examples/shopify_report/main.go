package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/gocolly/colly/v2"
)

func main() {
	// Get the domain name from the command-line argument
	if len(os.Args) < 2 {
		log.Fatal("Please provide the domain name as an argument.")
	}
	rootURL := "https://" + os.Args[1] // Assuming the user gives only the domain name (without 'https://')

	// Ask the user if the website is Shopify
	var isShopify string
	fmt.Print("Is it a Shopify website? (yes/no): ")
	fmt.Scanln(&isShopify)

	// Initialize a new Colly collector
	c := colly.NewCollector(
		colly.AllowedDomains(os.Args[1]),
	)

	// SEO Report Data
	report := make(map[string][]string)

	// Crawl all product pages and run SEO checks
	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		link := e.Attr("href")
		// Only visit product pages (check URL pattern)
		if isProductPage(link) {
			// Visit product page
			e.Request.Visit(link)
		} else if isCollectionPage(link) {
			// Visit collection pages for crawling all products
			e.Request.Visit(link)
		}
	})

	// Callback to check SEO issues on each page
	c.OnHTML("html", func(e *colly.HTMLElement) {
		// Run SEO checks on product pages
		if isProductPage(e.Request.URL.String()) {
			// SEO Check: Title and Meta Description
			checkTitleMeta(e, report)
			// SEO Check: Header Tags
			checkHeaders(e, report)
			// SEO Check: Schema Markup (JSON-LD)
			checkProductSchema(e, report)
			// SEO Check: Image Alt Text
			checkImageAltTags(e, report)
			// SEO Check: Canonical Tag
			checkCanonicalTag(e, report)
			// SEO Check: Product Reviews
			checkProductReviews(e, report)
		}

		// Shopify-specific checks (if applicable)
		if strings.ToLower(isShopify) == "yes" {
			checkShopifySpecifics(e, report)
		}
	})

	// Start crawling from the root URL
	err := c.Visit(rootURL)
	if err != nil {
		log.Fatal(err)
	}

	// Generate SEO report
	generateReport(report)
}

// Check if the URL looks like a product page
func isProductPage(url string) bool {
	return strings.Contains(url, "/products/")
}

// Check if the URL is a collection page
func isCollectionPage(url string) bool {
	return strings.Contains(url, "/collections/")
}

// Check Title Tag and Meta Description
func checkTitleMeta(e *colly.HTMLElement, report map[string][]string) {
	title := e.ChildText("title")
	if title == "" {
		report["Missing Title Tag"] = append(report["Missing Title Tag"], e.Request.URL.String())
	} else if len(title) < 50 {
		report["Short Title Tag"] = append(report["Short Title Tag"], e.Request.URL.String())
	}

	metaDesc := e.ChildAttr("meta[name=description]", "content")
	if metaDesc == "" {
		report["Missing Meta Description"] = append(report["Missing Meta Description"], e.Request.URL.String())
	} else if len(metaDesc) < 100 {
		report["Short Meta Description"] = append(report["Short Meta Description"], e.Request.URL.String())
	}
}

// Check Header Tags (<h1>, <h2>, etc.)
func checkHeaders(e *colly.HTMLElement, report map[string][]string) {
	// Shopify product pages may have h1 wrapped around the product title, but the selector may vary
	// Modify this if the theme uses a different structure
	h1Text := e.ChildText("h1.product-title")
	if h1Text == "" {
		// Also check if the h1 is within another element, like product__title or similar
		h1Text = e.ChildText("h1")
	}

	if h1Text == "" {
		report["Missing H1 Tag"] = append(report["Missing H1 Tag"], e.Request.URL.String())
	}

	// Check for h2, h3, and other headings to ensure content hierarchy
	h2Text := e.ChildText("h2")
	if h2Text == "" {
		report["Missing H2 Tag"] = append(report["Missing H2 Tag"], e.Request.URL.String())
	}
}

// Check for Product Schema (JSON-LD)
func checkProductSchema(e *colly.HTMLElement, report map[string][]string) {
	jsonLD := e.ChildText("script[type='application/ld+json']")
	if jsonLD == "" {
		report["Missing Product Schema"] = append(report["Missing Product Schema"], e.Request.URL.String())
	}
}

// Check for missing or poorly set image alt attributes
func checkImageAltTags(e *colly.HTMLElement, report map[string][]string) {
	e.ForEach("img", func(i int, el *colly.HTMLElement) {
		alt := el.Attr("alt")
		if alt == "" {
			report["Missing Alt Attribute"] = append(report["Missing Alt Attribute"], e.Request.URL.String())
		}
	})
}

// Check for Canonical Tag
func checkCanonicalTag(e *colly.HTMLElement, report map[string][]string) {
	canonicalURL := e.ChildAttr("link[rel='canonical']", "href")
	if canonicalURL == "" {
		report["Missing Canonical Tag"] = append(report["Missing Canonical Tag"], e.Request.URL.String())
	}
}

// Check for Product Reviews
func checkProductReviews(e *colly.HTMLElement, report map[string][]string) {
	reviews := e.ChildText(".product-reviews")
	if reviews == "" {
		report["Missing Product Reviews"] = append(report["Missing Product Reviews"], e.Request.URL.String())
	}
}

// Shopify-specific checks (if applicable)
func checkShopifySpecifics(e *colly.HTMLElement, report map[string][]string) {
	// Check if Shopify theme meta tags are present
	shopifyMetaTag := e.ChildAttr("meta[name='shopify-digital-downloads']", "content")
	if shopifyMetaTag == "" {
		report["Missing Shopify Meta Tag"] = append(report["Missing Shopify Meta Tag"], e.Request.URL.String())
	}

	// Shopify-specific schema check (product schema)
	shopifySchema := e.ChildText("script[type='application/json']")
	if shopifySchema == "" {
		report["Missing Shopify Product Schema"] = append(report["Missing Shopify Product Schema"], e.Request.URL.String())
	}
}

// Generate SEO report
func generateReport(report map[string][]string) {
	fmt.Println("SEO Report:")
	for issue, pages := range report {
		fmt.Printf("\n%s:\n", issue)
		for _, page := range pages {
			fmt.Printf("  - %s\n", page)
		}
	}
}
