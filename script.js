// --- 1. CONFIGURATION ---
const CATEGORIES = [
    'Politics', 'Sports', 'Entertainment', 'Health', 
    'Crime', 'Tech', 'Travel', 'Education', 
    'Fashion', 'Bhakti', 'Auto', 'International'
];
const HISTORY_KEY = 'marathi_categorizer_history';
// Flask backend (see backend/app.py). Change this if you run the API on a
// different host/port.
const API_BASE_URL = 'http://127.0.0.1:5000';
let history = [];

// --- 2. UI REFERENCES ---
const inputElement = document.getElementById('news-headline-input');
const categorizeButton = document.getElementById('categorize-button');
const loadingSpinner = document.getElementById('loading-spinner');
const resultsContainer = document.getElementById('results-container');
const predictedCategoryElement = document.getElementById('predicted-category');
const primaryConfidenceElement = document.getElementById('primary-confidence');
const reasoningElement = document.getElementById('reasoning');
const preprocessedTextElement = document.getElementById('preprocessed-text');
const messageArea = document.getElementById('message-area');
const topPredictionsList = document.getElementById('top-predictions-list');
const historyList = document.getElementById('history-list');
const noHistoryMessage = document.getElementById('no-history-message');


// --- 3. UTILITY FUNCTIONS ---

/**
 * Displays error messages in a visible UI area.
 */
function displayError(message) {
    messageArea.className = 'text-center p-3 rounded-xl mb-6 bg-red-100 text-red-700 border border-red-400';
    messageArea.textContent = `❌ Error: ${message}`;
    messageArea.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
}

/**
 * Calls the real Flask + TF-IDF/Naive Bayes backend (backend/app.py).
 * Returns the same { top_predictions, reasoning_english } shape that the
 * UI-rendering code below expects.
 */
async function categorizeViaBackend(headline) {
    const response = await fetch(`${API_BASE_URL}/api/categorize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ headline }),
    });

    if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.error || `Server responded with ${response.status}`);
    }

    return response.json();
}

/**
 * OFFLINE FALLBACK ONLY: a simple keyword-matching simulation, used when the
 * real backend (backend/app.py) can't be reached, so the UI still has
 * something to show. Results from this path are clearly labelled as
 * simulated/offline in the reasoning text below.
 */
function simulateCategorization(headline) {
    const normalizedHeadline = headline.toLowerCase();
    let primaryCategory = 'International'; // Default fallback, now moved to the end
    let reasoning = 'The headline content does not strongly align with a primary domestic category.';

    // --- ENHANCED KEYWORD-BASED CATEGORIZATION ---

    // 1. POLITICS: सरकार, मुख्यमंत्री, निवडणूक, विधानसभा, पक्ष, मंत्री, कायदा, राजकारण
    if (normalizedHeadline.includes('सरकार') || normalizedHeadline.includes('मुख्यमंत्री') || normalizedHeadline.includes('निवडणूक') || normalizedHeadline.includes('विधानसभा') || normalizedHeadline.includes('मंत्री') || normalizedHeadline.includes('पक्ष') || normalizedHeadline.includes('कायदा') || normalizedHeadline.includes('राजकारण')) {
        primaryCategory = 'Politics';
        reasoning = 'Words referring to governing bodies, leaders (Minister/CM), elections, or political parties indicate a focus on political affairs.';
    } 
    
    // 2. SPORTS: सामना, खेळ, क्रिकेट, फुटबॉल, कबड्डी, टेनिस, जिंकला, स्पर्धा, चॅम्पियन, गोल, धाव, मैदान
    else if (normalizedHeadline.includes('सामना') || normalizedHeadline.includes('खेळ') || normalizedHeadline.includes('क्रिकेट') || normalizedHeadline.includes('फुटबॉल') || normalizedHeadline.includes('कबड्डी') || normalizedHeadline.includes('टेनिस') || normalizedHeadline.includes('जिंकला') || normalizedHeadline.includes('स्पर्धा') || normalizedHeadline.includes('चॅम्पियन') || normalizedHeadline.includes('गोल') || normalizedHeadline.includes('धाव') || normalizedHeadline.includes('मैदान')) {
        primaryCategory = 'Sports';
        reasoning = 'The headline contains words strongly associated with competitive athletics, such as specific sports names, "match," "win," or "competition."';
    } 
    
    // 3. ENTERTAINMENT: अभिनेता, चित्रपट, गाणी, नाट्य, सिनेमा, कलाकार, प्रदर्शन, रिलीज, खाद्य, पदार्थ, खातो, खाता (Lifestyle/Food)
    else if (normalizedHeadline.includes('अभिनेता') || normalizedHeadline.includes('चित्रपट') || normalizedHeadline.includes('गाणी') || normalizedHeadline.includes('नाट्य') || normalizedHeadline.includes('सिनेमा') || normalizedHeadline.includes('कलाकार') || normalizedHeadline.includes('प्रदर्शन') || normalizedHeadline.includes('रिलीज') || normalizedHeadline.includes('खाद्य') || normalizedHeadline.includes('पदार्थ') || normalizedHeadline.includes('खातो') || normalizedHeadline.includes('खाता')) {
        primaryCategory = 'Entertainment';
        reasoning = 'Keywords like "actor," "movie," "songs," or food/eating terms clearly point toward the arts, film industry, or general lifestyle news.';
    } 
    
    // 4. HEALTH: डॉक्टर, आरोग्य, उपचार, हॉस्पिटल, दवाखाना, रोग, लस, आजार
    else if (normalizedHeadline.includes('डॉक्टर') || normalizedHeadline.includes('आरोग्य') || normalizedHeadline.includes('उपचार') || normalizedHeadline.includes('हॉस्पिटल') || normalizedHeadline.includes('दवाखाना') || normalizedHeadline.includes('रोग') || normalizedHeadline.includes('लस') || normalizedHeadline.includes('आजार')) {
        primaryCategory = 'Health';
        reasoning = 'The presence of terms like "doctor," "health," "treatment," or "hospital" categorizes this as a health story.';
    } 
    
    // 5. CRIME: गुन्हा, पोलिस, चोरी, अटक, खून, दरोडा, आरोपी, जेल
    else if (normalizedHeadline.includes('गुन्हा') || normalizedHeadline.includes('पोलिस') || normalizedHeadline.includes('चोरी') || normalizedHeadline.includes('अटक') || normalizedHeadline.includes('खून') || normalizedHeadline.includes('दरोडा') || normalizedHeadline.includes('आरोपी') || normalizedHeadline.includes('जेल')) {
        primaryCategory = 'Crime';
        reasoning = 'Keywords related to law enforcement, theft, murder, or arrest categorize this as a crime story.';
    } 
    
    // 6. TECH: तंत्रज्ञान, नवीन, इंटरनेट, मोबाईल, अॅप, संगणक, आयफोन, डेटा, एआय (AI), फोन (Also handles Stock Market/Finance)
    else if (normalizedHeadline.includes('तंत्रज्ञान') || normalizedHeadline.includes('नवीन') || normalizedHeadline.includes('इंटरनेट') || normalizedHeadline.includes('मोबाईल') || normalizedHeadline.includes('अॅप') || normalizedHeadline.includes('संगणक') || normalizedHeadline.includes('आयफोन') || normalizedHeadline.includes('डेटा') || normalizedHeadline.includes('एआय') || normalizedHeadline.includes('फोन') || normalizedHeadline.includes('शेअर') || normalizedHeadline.includes('बाजार') || normalizedHeadline.includes('गुंतवणूक') || normalizedHeadline.includes('सेन्सेक्स') || normalizedHeadline.includes('निफ्टी')) {
        primaryCategory = 'Tech';
        reasoning = 'References to "technology," "internet," "mobile," or **financial/stock market terms** classify this under the technology or business sector.';
    } 
    
    // 7. TRAVEL: पर्यटन, प्रवास, विमान, ट्रेन, बस, सुट्टी, ठिकाण, टूर
    else if (normalizedHeadline.includes('पर्यटन') || normalizedHeadline.includes('प्रवास') || normalizedHeadline.includes('विमान') || normalizedHeadline.includes('ट्रेन') || normalizedHeadline.includes('बस') || normalizedHeadline.includes('सुट्टी') || normalizedHeadline.includes('ठिकाण') || normalizedHeadline.includes('टूर')) {
        primaryCategory = 'Travel';
        reasoning = 'Words like "tourism," "travel," or references to modes of transport (plane, train) indicate a focus on travel news.';
    } 
    
    // 8. EDUCATION: शिक्षण, शाळा, कॉलेज, परीक्षा, निकाल, विद्यार्थी, अभ्यासक्रम
    else if (normalizedHeadline.includes('शिक्षण') || normalizedHeadline.includes('शाळा') || normalizedHeadline.includes('कॉलेज') || normalizedHeadline.includes('परीक्षा') || normalizedHeadline.includes('निकाल') || normalizedHeadline.includes('विद्यार्थी') || normalizedHeadline.includes('अभ्यासक्रम')) {
        primaryCategory = 'Education';
        reasoning = 'Terms referring to learning, schools, exams, or students are classified as education news.';
    } 
    
    // 9. FASHION: फॅशन, ड्रेस, स्टाईल, सौंदर्य, कपडे, सौंदर्यप्रसाधने
    else if (normalizedHeadline.includes('फॅशन') || normalizedHeadline.includes('ड्रेस') || normalizedHeadline.includes('स्टाईल') || normalizedHeadline.includes('सौंदर्य') || normalizedHeadline.includes('कपडे') || normalizedHeadline.includes('सौंदर्यप्रसाधने')) {
        primaryCategory = 'Fashion';
        reasoning = 'The headline contains words like "fashion," "style," or "beauty products."';
    } 
    
    // 10. BHAKTI: मंदिर, देव, पूजा, उत्सव, भजन, तीर्थक्षेत्र, आरती
    else if (normalizedHeadline.includes('मंदिर') || normalizedHeadline.includes('देव') || normalizedHeadline.includes('पूजा') || normalizedHeadline.includes('उत्सव') || normalizedHeadline.includes('भजन') || normalizedHeadline.includes('तीर्थक्षेत्र') || normalizedHeadline.includes('आरती')) {
        primaryCategory = 'Bhakti';
        reasoning = 'Keywords related to religion, temples, worship, or festivals are classified as Bhakti/Spiritual news.';
    } 
    
    // 11. AUTO: कार, बाईक, वाहन, गाडी, इंजिन, मायलेज, ऑटो, ड्रायव्हिंग
    else if (normalizedHeadline.includes('कार') || normalizedHeadline.includes('बाईक') || normalizedHeadline.includes('वाहन') || normalizedHeadline.includes('गाडी') || normalizedHeadline.includes('इंजिन') || normalizedHeadline.includes('मायलेज') || normalizedHeadline.includes('ऑटो') || normalizedHeadline.includes('ड्रायव्हिंग')) {
        primaryCategory = 'Auto';
        reasoning = 'References to vehicles (car, bike) or driving terms indicate the Auto sector.';
    } 
    
    // 12. INTERNATIONAL: आंतरराष्ट्रीय, अमेरिका, चीन, युद्ध, परदेश, रशिया, जगात (Also handles Weather/Environment)
    else if (normalizedHeadline.includes('आंतरराष्ट्रीय') || normalizedHeadline.includes('अमेरिका') || normalizedHeadline.includes('चीन') || normalizedHeadline.includes('युद्ध') || normalizedHeadline.includes('परदेश') || normalizedHeadline.includes('रशिया') || normalizedHeadline.includes('जगात') || normalizedHeadline.includes('हवामान') || normalizedHeadline.includes('पाऊस') || normalizedHeadline.includes('थंडी') || normalizedHeadline.includes('उष्णता')) {
        primaryCategory = 'International';
        reasoning = 'The headline explicitly mentions foreign countries, global affairs, or is related to **general environmental/weather updates** (which are non-domestic/non-political general news).';
    }
    
    // --- GENERAL/MISCELLANEOUS CATCH-ALL ---
    // If no category is clearly matched, default to the most common news categories (Politics or Entertainment)
    else if (headline.length > 5) { // Ensure it's not just a blank entry
        const genericKeywords = ['बातमी', 'चर्चेत', 'काय', 'घडले']; // "News", "in discussion", "what", "happened"
        
        if (genericKeywords.some(k => normalizedHeadline.includes(k))) {
            // Simple headlines without strong keywords often relate to Political or General News.
            primaryCategory = 'Politics';
            reasoning = 'Headline uses general news terms and lacks strong keywords, defaulting to the Politics category (common general news topic).';
        } else {
            // Simple statements (like food/personal) fit best here.
            primaryCategory = 'Entertainment';
            reasoning = 'The headline contains generic terms (e.g., simple statements) that best fit into general lifestyle or Entertainment news, lacking specific political or sports keywords.';
        }
    }


    // Generate simulated top predictions (primary category 90%, split the rest)
    const otherCategories = CATEGORIES.filter(c => c !== primaryCategory);
    
    // Shuffle and pick two secondary categories
    const shuffledOthers = otherCategories.sort(() => 0.5 - Math.random());
    const secondary1 = shuffledOthers[0];
    const secondary2 = shuffledOthers[1];

    return {
        top_predictions: [
            { category: primaryCategory, score: 0.90 + Math.random() * 0.05 }, // 90-95%
            { category: secondary1, score: 0.05 - Math.random() * 0.02 }, // 3-5%
            { category: secondary2, score: 0.01 + Math.random() * 0.01 } // 1-2%
        ].sort((a, b) => b.score - a.score), // Re-sort just to be safe
        reasoning_english: `[Offline demo mode — backend unreachable] ${reasoning}`
    };
}


// --- 5. HISTORY MANAGEMENT ---

/**
 * Loads history from localStorage and renders it.
 */
function loadHistory() {
    try {
        const storedHistory = localStorage.getItem(HISTORY_KEY);
        if (storedHistory) {
            history = JSON.parse(storedHistory);
        }
    } catch (e) {
        console.error("Could not load history from localStorage:", e);
        history = [];
    }
    renderHistory();
}

/**
 * Saves the current history array to localStorage.
 */
function saveHistory() {
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
        console.error("Could not save history to localStorage:", e);
    }
}

/**
 * Renders the history list in the UI.
 */
function renderHistory() {
    historyList.innerHTML = ''; // Clear existing items

    if (history.length === 0) {
        noHistoryMessage.classList.remove('hidden');
        historyList.appendChild(noHistoryMessage);
        return;
    }
    noHistoryMessage.classList.add('hidden');

    // Iterate over history items (newest first)
    history.slice().reverse().forEach((item, index) => {
        const historyItem = document.createElement('div');
        historyItem.className = 'bg-white p-4 rounded-xl shadow-md border border-gray-200 hover:shadow-lg transition duration-150 cursor-pointer';
        historyItem.onclick = () => {
            inputElement.value = item.headline; // Allow user to click to re-classify
            handleCategorization();
        };
        
        // Format timestamp
        const date = new Date(item.timestamp);
        const timeString = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        const dateString = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

        historyItem.innerHTML = `
            <div class="flex justify-between items-start border-b pb-2 mb-2">
                <p class="text-xl font-bold text-teal-700">${item.category}</p>
                <span class="text-xs text-gray-500 flex flex-col items-end">
                    <span>${dateString}</span>
                    <span class="font-mono">${timeString}</span>
                </span>
            </div>
            <p class="text-sm text-gray-800">${item.headline}</p>
        `;
        historyList.appendChild(historyItem);
    });
}

/**
 * Clears the history from local storage and the UI.
 */
function clearHistory() {
    // CRITICAL: Use a confirmation before deleting history
    // NOTE: We cannot use window.confirm() in the Canvas, but since this is for local VS Code, it's fine.
    if (!confirm("Are you sure you want to clear all classification history?")) {
        return;
    }

    history = [];
    saveHistory();
    renderHistory();
    messageArea.className = 'text-center p-3 rounded-xl mb-6 bg-yellow-100 text-yellow-700 border border-yellow-400';
    messageArea.textContent = 'History cleared successfully!';
    messageArea.classList.remove('hidden');
}

// Attach clearHistory globally (used by the button in index.html)
window.clearHistory = clearHistory;


// --- 4. CORE CATEGORIZATION LOGIC (SIMULATED) ---

async function handleCategorization() {
    const headline = inputElement.value.trim();

    if (!headline) {
        displayError("Please enter a Marathi headline.");
        return;
    }
    
    // Start loading state
    categorizeButton.disabled = true;
    loadingSpinner.classList.remove('hidden');
    messageArea.classList.remove('hidden');
    messageArea.className = 'text-center p-3 rounded-xl mb-6 bg-blue-100 text-blue-700';
    messageArea.textContent = 'Categorizing headline...'; // Updated text
    resultsContainer.classList.add('hidden');

    try {
        let result;
        try {
            // Real model: TF-IDF + Multinomial Naive Bayes served by Flask.
            result = await categorizeViaBackend(headline);
        } catch (backendError) {
            console.warn('Backend unreachable, falling back to offline simulation:', backendError);
            result = simulateCategorization(headline);
        }

        if (result.top_predictions && result.top_predictions.length > 0) {
            const topPrediction = result.top_predictions[0];

            // 1. Update Primary Prediction Panel
            predictedCategoryElement.textContent = topPrediction.category;
            primaryConfidenceElement.textContent = `${(topPrediction.score * 100).toFixed(2)}%`;
            reasoningElement.textContent = result.reasoning_english;
            
            if (result.preprocessed_text) {
                preprocessedTextElement.textContent = result.preprocessed_text;
                preprocessedTextElement.parentElement.classList.remove('hidden');
            } else {
                preprocessedTextElement.parentElement.classList.add('hidden');
            }

            // 2. Update Top Predictions List
            topPredictionsList.innerHTML = ''; // Clear previous list
            result.top_predictions.forEach((p, index) => {
                const li = document.createElement('li');
                li.className = `flex justify-between items-center text-sm ${index === 0 ? 'font-bold text-teal-700' : 'text-gray-600'}`;
                li.innerHTML = `
                    <span>${index + 1}. ${p.category}</span>
                    <span class="${index === 0 ? 'bg-teal-200 text-teal-800' : 'bg-gray-200 text-gray-700'} px-2 py-0.5 rounded-full text-xs font-medium">
                        ${(p.score * 100).toFixed(2)}%
                    </span>
                `;
                topPredictionsList.appendChild(li);
            });

            // 3. Add to History
            history.push({
                headline: headline,
                category: topPrediction.category,
                timestamp: new Date().toISOString()
            })

            // Keep history limited to the last 10 entries
            if (history.length > 10) {
                history.shift();
            }

            saveHistory();
            renderHistory();

            // Show results and hide messages
            resultsContainer.classList.remove('hidden');
            messageArea.classList.add('hidden');

        } else {
            displayError('Categorization logic failed to generate predictions.');
        }

    } catch (error) {
        console.error('Categorization Error:', error);
        displayError(`An internal error occurred during categorization: ${error.message}`);
    } finally {
        // End loading state
        categorizeButton.disabled = false;
        loadingSpinner.classList.add('hidden');
    }
}

// Attach handleCategorization globally (used by the button in index.html)
window.handleCategorization = handleCategorization;


// --- INITIALIZATION ---
loadHistory();
