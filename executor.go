package classifier

import (
	"fmt"
	"sync"
)

const (
	ErrClassNotFound = "[AstrA] Не удалось найти класс"
	ErrNotClassified = "[AstrA] Не удалось определить класс текста"
	ErrEmptyText     = "[AstrA] Передан пустой текст"
)

// Classifier описывает структуру классификатора
type Classifier struct {
	MapTokenToCategory map[string]map[string]int
	CategoriesCount    map[string]int
	tokenizer          Tokenizer
	mu                 sync.RWMutex
}

// Option описывает дополнительный настройки классификатора
type Option func(c *Classifier) error

// New создает новый классификатор со стандартным токенизатором
func New(opts ...Option) *Classifier {
	c := &Classifier{
		MapTokenToCategory: make(map[string]map[string]int),
		CategoriesCount:    make(map[string]int),
		tokenizer:          NewTokenizer(),
	}

	for _, opt := range opts {
		opt(c)
	}
	return c
}

// TokenizerCustom перезаписывает стандартный токенизатор
func TokenizerCustom(t Tokenizer) Option {
	return func(c *Classifier) error {
		c.tokenizer = t
		return nil
	}
}

// Train производит обучение на категорию category по данным документа trainData
func (c *Classifier) Train(trainData string, category string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for t := range c.tokenizer.Tokenize(trainData) {
		c.addToken(t, category)
	}

	c.addCategory(category)
	return nil
}

// Classify производит попытку определить класс переданного текста в документа testData
func (c *Classifier) Classify(data string) (guessedClass string, coincidenceIndex float64, err error) {
	if data == "" {
		return "", 0.0, fmt.Errorf(ErrEmptyText)
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	// Определение максимального индекса совпадения
	maxCoincidenceIndex := 0.0
	for _, cat := range c.getModelCategories() {
		catCI := c.getProb(data, cat)
		if catCI > maxCoincidenceIndex {
			maxCoincidenceIndex = catCI
			guessedClass = cat
		}
	}

	if guessedClass == "" {
		return "", 0.0, fmt.Errorf(ErrNotClassified)
	}
	return guessedClass, maxCoincidenceIndex, nil
}

// GetProb определяет индекс совпадения str с категориями модели и выдает лучший результат
func (c *Classifier) GetProb(str string) (map[string]float64, string) {
	probs := make(map[string]float64)

	c.mu.RLock()
	defer c.mu.RUnlock()

	bestProb := 0.0
	var respCat string
	for _, cat := range c.getModelCategories() {
		prob := c.getProb(str, cat)
		if prob > 0 {
			probs[cat] = prob
		}

		if prob > bestProb {
			bestProb = prob
			respCat = cat
		}
	}

	return probs, respCat
}

// addToken добавляет новый токен в категорию
func (c *Classifier) addToken(t string, cat string) {
	if _, ok := c.MapTokenToCategory[t]; !ok {
		c.MapTokenToCategory[t] = make(map[string]int)
	}
	c.MapTokenToCategory[t][cat]++
}

// countTokensInCategory возвращает вес token в категории
func (c *Classifier) countTokenInCategory(token string, category string) float64 {
	if _, ok := c.MapTokenToCategory[token]; ok {
		return float64(c.MapTokenToCategory[token][category])
	}
	return 0.0
}

// calcTokenWeight возвращает общий вес токена в классификаторе
func (c *Classifier) calcTokenWeight(token string) float64 {
	var weight float64
	for _, cat := range c.getModelCategories() {
		weight += float64(c.MapTokenToCategory[token][cat])
	}

	// При отсутствии токена в классификаторе, возвращается минимальный вес
	if weight > 0 {
		return weight
	}
	return 0.001
}

// addCategory добавляет в модель новую категорию cat
func (c *Classifier) addCategory(cat string) {
	c.CategoriesCount[cat]++
}

// categoryTokensCount возвращает количество токенов в category
func (c *Classifier) categoryTokensCount(cat string) float64 {
	if _, ok := c.CategoriesCount[cat]; ok {
		return float64(c.CategoriesCount[cat])
	}
	return 0.0
}

// countOverallTokens возращает общее количество токенов в модели
func (c *Classifier) countOverallTokens() int {
	sum := 0
	for _, v := range c.CategoriesCount {
		sum += v
	}
	return sum
}

// getModelCategories возвращает общий список категорий в модели
func (c *Classifier) getModelCategories() []string {
	var keys []string
	for k := range c.CategoriesCount {
		keys = append(keys, k)
	}
	return keys
}

// getTokenProb возвращает вероятность отношения токена к категории
func (c *Classifier) getTokenProb(token string, category string) float64 {
	if c.categoryTokensCount(category) == 0 {
		return 0.0
	}

	return c.countTokenInCategory(token, category) / c.categoryTokensCount(category)
}

// getProb получает вероятность принадлежности токена к категории с учетом их относительных весов
func (c *Classifier) getProb(data string, category string) float64 {
	// categoryProb := c.categoryTokensCount(category) / float64(c.countOverallTokens())
	categoryProb := float64(1 / len(c.CategoriesCount))
	docProb := c.getTextProb(data, category)
	return docProb * categoryProb
}

// getDocumentProb определяет вероятность отношения текста data к cat
func (c *Classifier) getTextProb(data string, cat string) float64 {
	prob := 1.0
	for t := range c.tokenizer.Tokenize(data) {
		prob *= c.getWeightedProb(t, cat)
	}
	return prob
}

// getWeightedProb определяет взвешенную вероятность отношения токена к категории
func (c *Classifier) getWeightedProb(token string, cat string) float64 {
	sum := 0.0
	prob := c.getTokenProb(token, cat)
	for _, category := range c.getModelCategories() {
		sum += c.countTokenInCategory(token, category)
	}

	result := ((c.calcTokenWeight(token) * 1 / float64(len(c.CategoriesCount))) + (sum * prob)) / (1.0 + sum)
	return result
}
