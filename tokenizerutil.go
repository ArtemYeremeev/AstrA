package classifier

const defaultBufferSize = 50

// Predicate описывает функцию предиката
type Predicate func(string) bool

// Mapper описывает функцию-маппер
type Mapper func(string) string

// Map применяет функцию f к каждому элементу входного канала
func Map(vs chan string, f ...Mapper) chan string {
	stream := make(chan string, defaultBufferSize)

	go func() {
		for v := range vs {
			for _, fn := range f {
				v = fn(v)
			}
			stream <- v
		}
		close(stream)
	}()

	return stream
}

// Filter применяет функции фильтров к каждому элементу входного потока
func Filter(vs chan string, filters ...Predicate) chan string {
	stream := make(chan string, defaultBufferSize)
	apply := func(text string) bool {
		for _, f := range filters {
			if !f(text) {
				return false
			}
		}
		return true
	}

	go func() {
		for text := range vs {
			if apply(text) {
				stream <- text
			}
		}
		close(stream)
	}()

	return stream
}
