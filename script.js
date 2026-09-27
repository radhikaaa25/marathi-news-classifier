// ============================================================
//  Marathi News Classifier – Frontend Script
//  Includes: step-by-step pipeline display, offline fallback,
//  history, sample chips.
// ============================================================

// --- CONFIG ---
const API_BASE_URL = 'http://127.0.0.1:5000';
const HISTORY_KEY  = 'marathi_categorizer_history';
const CATEGORIES   = [
    'Politics','Sports','Entertainment','Health',
    'Crime','Tech','Travel','Education',
    'Fashion','Bhakti','Auto','International'
];

let history = [];

// --- DOM REFS ---
const inputEl           = document.getElementById('news-headline-input');
const classifyBtn       = document.getElementById('categorize-button');
const btnText           = document.getElementById('button-text');
const spinner           = document.getElementById('loading-spinner');
const btnArrow          = document.querySelector('.btn-arrow');
const messageArea       = document.getElementById('message-area');
const pipelineSection   = document.getElementById('pipeline-section');
const resultsContainer  = document.getElementById('results-container');
const predictedCatEl    = document.getElementById('predicted-category');
const confidenceEl      = document.getElementById('primary-confidence');
const confidenceBar     = document.getElementById('confidence-bar');
const reasoningEl       = document.getElementById('reasoning');
const topListEl         = document.getElementById('top-predictions-list');
const historyList       = document.getElementById('history-list');
const noHistoryMsg      = document.getElementById('no-history-message');

// Pipeline output elements
const outRaw       = document.getElementById('out-raw');
const outNorm      = document.getElementById('out-normalized');
const outTokens    = document.getElementById('out-tokens');
const outNoPunct   = document.getElementById('out-no-punct');
const outNoStop    = document.getElementById('out-no-stopwords');
const outStemmed   = document.getElementById('out-stemmed');
const finalText    = document.getElementById('final-text');
const cntTokens    = document.getElementById('count-tokens');
const cntNoPunct   = document.getElementById('count-no-punct');
const cntNoStop    = document.getElementById('count-no-stopwords');
const cntStemmed   = document.getElementById('count-stemmed');

// ─────────────────────────────────────────────────
// UTILITY
// ─────────────────────────────────────────────────

function showMessage(text, type = 'info') {
    messageArea.textContent = text;
    messageArea.className = 'message-area ' + type;
    messageArea.classList.remove('hidden');
}

function hideMessage() { messageArea.classList.add('hidden'); }

function setLoading(on) {
    classifyBtn.disabled = on;
    spinner.classList.toggle('hidden', !on);
    btnText.textContent = on ? 'Classifying…' : 'Classify';
    if (btnArrow) btnArrow.style.display = on ? 'none' : '';
}

function useSample(btn) {
    inputEl.value = btn.textContent.trim();
    inputEl.focus();
    handleCategorization();
}
window.useSample = useSample;

// ─────────────────────────────────────────────────
// PIPELINE RENDERER
// ─────────────────────────────────────────────────

function makeChips(tokens, chipClass) {
    return tokens.map((t, i) => {
        const span = document.createElement('span');
        span.className = 'token-chip ' + chipClass;
        span.textContent = t;
        span.style.animationDelay = (i * 30) + 'ms';
        return span;
    });
}

function renderPipeline(steps) {
    // Step 1 – raw
    outRaw.textContent = steps.step1_raw || '(empty)';

    // Step 2 – normalized
    outNorm.textContent = steps.step2_normalized || '(empty)';

    // Step 3 – tokens (color-code punctuation vs. words)
    outTokens.innerHTML = '';
    const PUNCT_RE = /^[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u0964\u0965\u0970]+$/;
    steps.step3_tokens.forEach((t, i) => {
        const span = document.createElement('span');
        span.className = 'token-chip ' + (PUNCT_RE.test(t) ? 'chip-punct' : 'chip-word');
        span.textContent = t;
        span.style.animationDelay = (i * 25) + 'ms';
        outTokens.appendChild(span);
    });
    cntTokens.textContent = steps.step3_tokens.length + ' tokens';

    // Step 4 – no punct
    outNoPunct.innerHTML = '';
    makeChips(steps.step4_no_punct, 'chip-nopunct').forEach(c => outNoPunct.appendChild(c));
    const removed4 = steps.step3_tokens.length - steps.step4_no_punct.length;
    cntNoPunct.textContent = steps.step4_no_punct.length + ' tokens  (' + removed4 + ' removed)';

    // Step 5 – no stopwords
    outNoStop.innerHTML = '';
    makeChips(steps.step5_no_stopwords, 'chip-nostop').forEach(c => outNoStop.appendChild(c));
    const removed5 = steps.step4_no_punct.length - steps.step5_no_stopwords.length;
    cntNoStop.textContent = steps.step5_no_stopwords.length + ' tokens  (' + removed5 + ' stopwords removed)';

    // Step 6 – stemmed (show before→after for changed words)
    outStemmed.innerHTML = '';
    steps.step6_stemmed.forEach((stem, i) => {
        const orig = steps.step5_no_stopwords[i] || stem;
        const span = document.createElement('span');
        span.className = 'token-chip chip-stemmed';
        span.style.animationDelay = (i * 25) + 'ms';
        span.textContent = orig !== stem ? orig + ' → ' + stem : stem;
        span.title = orig !== stem ? 'Stemmed: ' + orig + ' → ' + stem : 'No change';
        outStemmed.appendChild(span);
    });
    const changed = steps.step5_no_stopwords.filter((w,i) => w !== steps.step6_stemmed[i]).length;
    cntStemmed.textContent = steps.step6_stemmed.length + ' tokens  (' + changed + ' stemmed)';

    // Final
    finalText.textContent = steps.step6_stemmed.join(' ') || '(empty after preprocessing)';

    // Reveal pipeline section
    pipelineSection.classList.remove('hidden');

    // Animate each step in with staggered delay
    document.querySelectorAll('.pipeline-step').forEach((el, i) => {
        el.classList.remove('visible');
        setTimeout(() => el.classList.add('visible'), 80 * i);
    });
}

// ─────────────────────────────────────────────────
// RESULTS RENDERER
// ─────────────────────────────────────────────────

function renderResults(result) {
    const top = result.top_predictions[0];
    predictedCatEl.textContent = top.category;
    const pct = (top.score * 100).toFixed(2);
    confidenceEl.textContent = pct + '%';
    setTimeout(() => { confidenceBar.style.width = pct + '%'; }, 80);
    reasoningEl.textContent = result.reasoning_english || '—';

    topListEl.innerHTML = '';
    result.top_predictions.forEach((p, i) => {
        const li = document.createElement('li');
        li.className = 'score-item';
        const isTop = i === 0;
        const barClass = i === 0 ? 'is-top' : (i === 1 ? 'is-2nd' : 'is-3rd');
        const pctN = (p.score * 100).toFixed(2);
        li.innerHTML =
            '<div class="score-item-top">' +
              '<span class="score-name' + (isTop ? ' is-top' : '') + '">' + (i+1) + '. ' + p.category + '</span>' +
              '<span class="score-pct' + (isTop ? ' is-top' : '') + '">' + pctN + '%</span>' +
            '</div>' +
            '<div class="score-bar-wrap"><div class="score-bar ' + barClass + '" data-w="' + pctN + '"></div></div>';
        topListEl.appendChild(li);
    });
    // Animate bars
    setTimeout(() => {
        topListEl.querySelectorAll('.score-bar').forEach(b => {
            b.style.width = b.dataset.w + '%';
        });
    }, 120);

    resultsContainer.classList.remove('hidden');
}

// ─────────────────────────────────────────────────
// OFFLINE SIMULATION (unchanged shape for offline mode)
// ─────────────────────────────────────────────────

function simulatePipeline(headline) {
    // Minimal client-side simulation when backend unreachable
    const raw = headline.trim();
    const tokens = raw.split(/[\s,।]+/).filter(Boolean);
    return {
        step1_raw: raw,
        step2_normalized: raw,
        step3_tokens: tokens,
        step4_no_punct: tokens,
        step5_no_stopwords: tokens,
        step6_stemmed: tokens,
    };
}

function simulateCategorization(headline) {
    const h = headline.toLowerCase();
    const map = [
        ['Politics',       ['सरकार','मुख्यमंत्री','निवडणूक','विधानसभा','मंत्री','पक्ष']],
        ['Sports',         ['सामना','क्रिकेट','फुटबॉल','कबड्डी','विजय','स्पर्धा']],
        ['Entertainment',  ['अभिनेता','चित्रपट','गाणी','सिनेमा','कलाकार','रिलीज']],
        ['Health',         ['डॉक्टर','आरोग्य','उपचार','हॉस्पिटल','रोग','लस']],
        ['Crime',          ['गुन्हा','पोलिस','चोरी','अटक','खून','दरोडा']],
        ['Tech',           ['तंत्रज्ञान','इंटरनेट','मोबाईल','अॅप','एआय','सॉफ्टवेअर']],
        ['Travel',         ['पर्यटन','प्रवास','विमान','ट्रेन','सुट्टी','टूर']],
        ['Education',      ['शिक्षण','शाळा','कॉलेज','परीक्षा','निकाल','विद्यार्थी']],
        ['Fashion',        ['फॅशन','ड्रेस','स्टाईल','सौंदर्य','कपडे']],
        ['Bhakti',         ['मंदिर','देव','पूजा','उत्सव','भजन','तीर्थक्षेत्र']],
        ['Auto',           ['कार','बाईक','वाहन','गाडी','इलेक्ट्रिक']],
        ['International',  ['आंतरराष्ट्रीय','अमेरिका','चीन','युद्ध','रशिया']],
    ];
    let primary = 'International';
    for (const [cat, kws] of map) {
        if (kws.some(k => h.includes(k))) { primary = cat; break; }
    }
    const others = CATEGORIES.filter(c => c !== primary).sort(() => Math.random()-0.5);
    return {
        top_predictions: [
            { category: primary,   score: 0.90 + Math.random()*0.05 },
            { category: others[0], score: 0.05 - Math.random()*0.02 },
            { category: others[1], score: 0.01 + Math.random()*0.01 },
        ].sort((a,b) => b.score - a.score),
        reasoning_english: '[Offline demo — backend unreachable] Keyword-based fallback used.',
        preprocessed_text: headline,
        preprocessing_steps: simulatePipeline(headline),
    };
}

// ─────────────────────────────────────────────────
// HISTORY
// ─────────────────────────────────────────────────

function loadHistory() {
    try { history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
    catch { history = []; }
    renderHistory();
}

function saveHistory() {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch {}
}

function renderHistory() {
    historyList.innerHTML = '';
    if (!history.length) {
        noHistoryMsg.classList.remove('hidden');
        historyList.appendChild(noHistoryMsg);
        return;
    }
    noHistoryMsg.classList.add('hidden');
    history.slice().reverse().forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.onclick = () => { inputEl.value = item.headline; handleCategorization(); };
        const d = new Date(item.timestamp);
        div.innerHTML =
            '<span class="history-headline">' + item.headline + '</span>' +
            '<span class="history-cat">' + item.category + '</span>' +
            '<span class="history-time">' +
                d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}) +
            '</span>';
        historyList.appendChild(div);
    });
}

function clearHistory() {
    if (!confirm('Clear all classification history?')) return;
    history = []; saveHistory(); renderHistory();
    showMessage('History cleared.', 'info');
}
window.clearHistory = clearHistory;

// ─────────────────────────────────────────────────
// MAIN HANDLER
// ─────────────────────────────────────────────────

async function handleCategorization() {
    const headline = inputEl.value.trim();
    if (!headline) { showMessage('⚠️ Please enter a Marathi headline.', 'error'); return; }

    setLoading(true);
    showMessage('Processing…', 'info');
    pipelineSection.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    confidenceBar.style.width = '0';

    try {
        let result;
        try {
            const resp = await fetch(API_BASE_URL + '/api/categorize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ headline }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.error || 'Server error ' + resp.status);
            }
            result = await resp.json();
        } catch (backendErr) {
            console.warn('Backend unreachable, using offline simulation:', backendErr);
            result = simulateCategorization(headline);
        }

        if (!result.top_predictions?.length) {
            showMessage('No predictions returned.', 'error'); return;
        }

        // Show preprocessing pipeline
        const steps = result.preprocessing_steps || simulatePipeline(headline);
        renderPipeline(steps);

        // Show classification result
        renderResults(result);

        // Save history
        history.push({ headline, category: result.top_predictions[0].category, timestamp: new Date().toISOString() });
        if (history.length > 10) history.shift();
        saveHistory(); renderHistory();

        hideMessage();

    } catch (err) {
        console.error(err);
        showMessage('Error: ' + err.message, 'error');
    } finally {
        setLoading(false);
    }
}
window.handleCategorization = handleCategorization;

// ─────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────
loadHistory();
