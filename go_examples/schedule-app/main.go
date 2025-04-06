package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"text/template"

	_ "modernc.org/sqlite"
)

type Task struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
	Done bool   `json:"done"`
	Days string `json:"days"` // e.g., "Mon,Wed,Fri"
}

var db *sql.DB

func initDB() {
	var err error
	db, err = sql.Open("sqlite", "./database.db") // Use global 'db'!
	if err != nil {
		panic(err)
	}

	// Create table if not exists
	query := `
	CREATE TABLE IF NOT EXISTS tasks (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		done BOOLEAN DEFAULT FALSE,
		days TEXT NOT NULL
	);`
	_, err = db.Exec(query)
	if err != nil {
		panic(err)
	}

	fmt.Println("Database initialized successfully")
}

func main() {
	initDB()
	defer db.Close()

	http.HandleFunc("/", serveIndex)
	http.HandleFunc("/tasks", handleTasks)
	http.HandleFunc("/add-task", addTask)
	http.HandleFunc("/toggle-task", toggleTask)
	http.HandleFunc("/delete-task", deleteTask)

	// Serve static files (CSS, JS)
	fs := http.FileServer(http.Dir("static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))

	fmt.Println("Server started at http://localhost:8080")
	http.ListenAndServe(":8080", nil)
}

// Serve the HTML page
func serveIndex(w http.ResponseWriter, r *http.Request) {
	tmpl, _ := template.ParseFiles("templates/index.html")
	tmpl.Execute(w, nil)
}

// Handle fetching tasks
func handleTasks(w http.ResponseWriter, r *http.Request) {
	if r.Method == "GET" {
		rows, err := db.Query("SELECT id, name, done, days FROM tasks")
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var tasks []Task
		for rows.Next() {
			var t Task
			err := rows.Scan(&t.ID, &t.Name, &t.Done, &t.Days)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			tasks = append(tasks, t)
		}

		if len(tasks) == 0 {
			w.Write([]byte("[]")) // Ensure we return an empty JSON array instead of null
			return
		}

		json.NewEncoder(w).Encode(tasks)
	}
}

// Handle adding a new task
func addTask(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		var t Task
		err := json.NewDecoder(r.Body).Decode(&t)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		days := strings.Split(t.Days, ",") // Split days into a slice

		for _, day := range days {
			_, err = db.Exec("INSERT INTO tasks (name, days, done) VALUES (?, ?, ?)", t.Name, day, false)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}

		w.WriteHeader(http.StatusCreated)
	}
}

// Helper function to split comma-separated days
func splitDays(days string) []string {
	var result []string
	for _, d := range strings.Split(days, ",") {
		trimmed := strings.TrimSpace(d)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

// Handle marking a task as done/undone
func toggleTask(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		var t Task
		json.NewDecoder(r.Body).Decode(&t)

		_, err := db.Exec("UPDATE tasks SET done = ? WHERE id = ?", !t.Done, t.ID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
	}
}

func deleteTask(w http.ResponseWriter, r *http.Request) {
	if r.Method == "DELETE" {
		idStr := r.URL.Query().Get("id")
		if idStr == "" {
			http.Error(w, "Missing task ID", http.StatusBadRequest)
			return
		}

		id, err := strconv.Atoi(idStr)
		if err != nil {
			http.Error(w, "Invalid task ID", http.StatusBadRequest)
			return
		}

		_, err = db.Exec("DELETE FROM tasks WHERE id = ?", id)
		if err != nil {
			http.Error(w, "Failed to delete task", http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)
	}
}
