package main

import "fmt"

func sum(a int, b int) int {
	return a + b
}

func even_or_odd(n int) string {
	if n%2 == 0 {
		return "even"
	} else {
		return "odd"
	}
}

type Person struct {
	Name string
	Age  int
}

func (P Person) Greet() {
	fmt.Printf("Hello, my name is %s and I am %d years old", P.Name, P.Age)
}

func main() {
	//fmt.Println("hello world")
	//fmt.Println(sum(2, 3))
	//fmt.Println("1:", even_or_odd(1))
	//fmt.Println("4:", even_or_odd(4))
	/*for i := 0; i < 10; i++ {
		fmt.Println(i)
	}*/
	/*numbers := []int{1, 2, 3, 4, 5}
	for _, i := range numbers {
		fmt.Println(i)
	}*/
	/*capitals := map[string]string{
		"France":  "Paris",
		"Canada":  "Ottawa",
		"Germany": "Berlin",
	}
	fmt.Println(capitals["France"])*/
	//person := Person{Name: "Henry", Age: 38}
	//person.Greet()

}
