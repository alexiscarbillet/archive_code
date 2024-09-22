package main

import (
	"fmt"
	"time"

	"github.com/shirou/gopsutil/cpu"
	"github.com/shirou/gopsutil/disk"
	"github.com/shirou/gopsutil/mem"
)

func main() {
	// Continuously monitor system metrics
	ticker := time.NewTicker(5 * time.Second) // Adjust the interval as needed
	defer ticker.Stop()

	for range ticker.C {
		// Retrieve system metrics
		cpuUsage, _ := cpu.Percent(time.Second, false)
		memInfo, _ := mem.VirtualMemory()
		diskInfo, _ := disk.Usage("/")

		// Print system metrics
		fmt.Printf("CPU Usage: %.2f%%\n", cpuUsage[0])
		fmt.Printf("Memory Usage: %.2f%%\n", memInfo.UsedPercent)
		fmt.Printf("Disk Usage: %.2f%%\n", diskInfo.UsedPercent)

		fmt.Println("-------------------------------------")
	}
}
