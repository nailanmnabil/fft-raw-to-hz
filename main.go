package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"strconv"

	"github.com/mjibson/go-dsp/fft"
)

type EEGData struct {
	Timestamp float64
	EEG       [6]float64
}

type ProcessedEEGData struct {
	Timestamp float64
	Delta     [6]float64
	Theta     [6]float64
	Alpha     [6]float64
	Beta      [6]float64
	Gamma     [6]float64
}

const (
	windowSize = 256 // 1 second of data at 256 Hz
	overlap    = 128 // 50% overlap
	sampleRate = 256.0
)

func main() {
	// Baca file CSV
	data := readCSV("eeg_data.csv")

	// Proses data untuk semua channel EEG
	processedData := processAllChannels(data)

	// Tulis hasil ke CSV
	writeCSV(processedData, "processed_eeg_data.csv")

	fmt.Println("Processing complete. Results written to processed_eeg_data.csv")
}

func readCSV(filename string) []EEGData {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	var data []EEGData
	for _, record := range records[1:] { // Skip header
		timestamp, _ := strconv.ParseFloat(record[0], 64)
		eegData := EEGData{Timestamp: timestamp}
		for i := 0; i < 6; i++ {
			eegData.EEG[i], _ = strconv.ParseFloat(record[i+1], 64)
		}
		data = append(data, eegData)
	}

	return data
}

func extractChannelData(data []EEGData, channel int) []float64 {
	channelData := make([]float64, len(data))
	for i, d := range data {
		channelData[i] = d.EEG[channel]
	}
	return channelData
}

func processAllChannels(data []EEGData) []ProcessedEEGData {
	var processedData []ProcessedEEGData

	for i := 0; i < len(data)-windowSize; i += overlap {
		processed := ProcessedEEGData{Timestamp: data[i].Timestamp}

		for channel := 0; channel < 6; channel++ {
			channelData := make([]float64, windowSize)
			for j := 0; j < windowSize; j++ {
				channelData[j] = data[i+j].EEG[channel]
			}

			windowed := applyHammingWindow(channelData)
			spectrum := fft.FFTReal(windowed)

			processed.Delta[channel] = powerToDb(sumPower(spectrum, 0.5, 4))
			processed.Theta[channel] = powerToDb(sumPower(spectrum, 4, 8))
			processed.Alpha[channel] = powerToDb(sumPower(spectrum, 8, 13))
			processed.Beta[channel] = powerToDb(sumPower(spectrum, 13, 30))
			processed.Gamma[channel] = powerToDb(sumPower(spectrum, 30, 100))
		}

		processedData = append(processedData, processed)
	}

	return processedData
}

func applyHammingWindow(data []float64) []float64 {
	windowed := make([]float64, len(data))
	for i := range data {
		windowed[i] = data[i] * (0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(len(data)-1)))
	}
	return windowed
}

func sumPower(spectrum []complex128, lowFreq, highFreq float64) float64 {
	lowBin := int(lowFreq * float64(windowSize) / sampleRate)
	highBin := int(highFreq * float64(windowSize) / sampleRate)
	power := 0.0
	for i := lowBin; i < highBin && i < len(spectrum); i++ {
		power += cmplx.Abs(spectrum[i]) * cmplx.Abs(spectrum[i])
	}
	return power / float64(windowSize)
}

func powerToDb(power float64) float64 {
	if power <= 0 {
		return -math.Inf(1)
	}
	return 10 * math.Log10(power)
}

func writeCSV(data []ProcessedEEGData, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := []string{
		"timestamps",
		"delta_1", "delta_2", "delta_3", "delta_4", "delta_5", "delta_6",
		"theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6",
		"alpha_1", "alpha_2", "alpha_3", "alpha_4", "alpha_5", "alpha_6",
		"beta_1", "beta_2", "beta_3", "beta_4", "beta_5", "beta_6",
		"gamma_1", "gamma_2", "gamma_3", "gamma_4", "gamma_5", "gamma_6",
	}
	if err := writer.Write(header); err != nil {
		panic(err)
	}

	// Write data
	for _, row := range data {
		record := []string{
			strconv.FormatFloat(row.Timestamp, 'f', -1, 64),
		}
		for i := 0; i < 6; i++ {
			record = append(record, strconv.FormatFloat(row.Delta[i], 'f', 2, 64))
		}
		for i := 0; i < 6; i++ {
			record = append(record, strconv.FormatFloat(row.Theta[i], 'f', 2, 64))
		}
		for i := 0; i < 6; i++ {
			record = append(record, strconv.FormatFloat(row.Alpha[i], 'f', 2, 64))
		}
		for i := 0; i < 6; i++ {
			record = append(record, strconv.FormatFloat(row.Beta[i], 'f', 2, 64))
		}
		for i := 0; i < 6; i++ {
			record = append(record, strconv.FormatFloat(row.Gamma[i], 'f', 2, 64))
		}

		if err := writer.Write(record); err != nil {
			panic(err)
		}
	}
}
