let allTasks = []; // Store all tasks globally

document.addEventListener("DOMContentLoaded", function () {
    fetchTasks();

    // Task form submission
    document.getElementById("taskForm").addEventListener("submit", function (event) {
        event.preventDefault();
        addTask();
    });

    // Filter by day
    document.getElementById("filterDays").addEventListener("change", function () {
        renderTasks(this.value);
    });
});

// Function to fetch tasks from the server
function fetchTasks() {
    fetch("/tasks")
        .then(response => response.json())
        .then(tasks => {
            allTasks = tasks; // Store fetched tasks
            renderTasks("all"); // Show all tasks initially
        })
        .catch(error => console.error("Error fetching tasks:", error));
}

// Function to render tasks based on selected filter
function renderTasks(selectedDay) {
    const taskList = document.getElementById("taskList");
    taskList.innerHTML = ""; // Clear existing tasks

    allTasks.forEach(task => {
        if (selectedDay === "all" || task.days.includes(selectedDay)) {
            const li = document.createElement("li");
            li.innerHTML = `
                ${task.name} (${task.days})
                <button class="delete-task" data-id="${task.id}">❌</button>
            `;
            taskList.appendChild(li);
        }
    });

    // Only need to attach delete event now
    document.querySelectorAll(".delete-task").forEach(button => {
        button.addEventListener("click", deleteTask);
    });
}

// Add new task
function addTask() {
    const taskName = document.getElementById("taskName").value;
    const taskDays = document.getElementById("taskDays").value; // Get selected day

    if (!taskName || !taskDays) {
        alert("Please enter a task name and select a day.");
        return;
    }

    const newTask = { name: taskName, days: taskDays };

    fetch("/add-task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newTask),
    })
    .then(response => {
        if (response.ok) {
            fetchTasks(); // Refresh tasks after adding
            document.getElementById("taskForm").reset();
        } else {
            console.error("Failed to add task");
        }
    })
    .catch(error => console.error("Error adding task:", error));
}

// Toggle task completion
function toggleTask(event) {
    const taskId = event.target.dataset.id;

    fetch("/toggle-task", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: taskId }),
    })
    .then(response => {
        if (response.ok) fetchTasks(); // Refresh tasks
    })
    .catch(error => console.error("Error toggling task:", error));
}

// Delete task
function deleteTask(event) {
    const taskId = event.target.dataset.id;

    fetch(`/delete-task?id=${taskId}`, { method: "DELETE" })
        .then(response => {
            if (response.ok) fetchTasks(); // Refresh tasks
        })
        .catch(error => console.error("Error deleting task:", error));
}
