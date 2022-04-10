package classifier

import (
	"bufio"
	"strings"
)

// Tokenizer содержит методы для обработки входных данных на токены
type Tokenizer interface {
	Tokenize(string) chan string
}

// Tokenize разбивает переданный документ на токены
func (t *StdTokenizer) Tokenize(data string) chan string {
	tokenizer := bufio.NewScanner(strings.NewReader(data))
	tokenizer.Split(bufio.ScanWords)
	tokens := make(chan string, t.bufferSize)

	go func() {
		for tokenizer.Scan() {
			tokens <- tokenizer.Text()
		}
		close(tokens)
	}()

	return t.pipeline(tokens)
}

// StdTokenizer содержит tokenizer для обработки входных данных по разделителю
type StdTokenizer struct {
	transforms []Mapper
	filters    []Predicate
	bufferSize int
}

// StdOption содержит настройки tokenizer
type StdOption func(*StdTokenizer)

// NewTokenizer создает новый экзмепляр tokenizer
func NewTokenizer(opts ...StdOption) *StdTokenizer {
	tokenizer := &StdTokenizer{
		bufferSize: 100,
		transforms: []Mapper{ // Токены приводятся в нижний регистр
			strings.ToLower,
		},
		filters: []Predicate{ // Токены фильтруются по массиву стоп-слов
			IsNotStopWord,
		},
	}

	for _, opt := range opts {
		opt(tokenizer)
	}

	return tokenizer
}

// pipeline применяет функции фильтров и мапперов на входной поток токенов
func (t *StdTokenizer) pipeline(in chan string) chan string {
	return Map(Filter(in, t.filters...), t.transforms...)
}

// BufferSize определяет размер входящего канала
func BufferSize(size int) StdOption {
	return func(t *StdTokenizer) {
		t.bufferSize = size
	}
}

// Transforms...
func Transforms(m ...Mapper) StdOption {
	return func(t *StdTokenizer) {
		t.transforms = m
	}
}

// Filters...
func Filters(f ...Predicate) StdOption {
	return func(t *StdTokenizer) {
		t.filters = f
	}
}
