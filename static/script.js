document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("summarizer-form");
  const textInput = document.getElementById("text-input");
  const fileInput = document.getElementById("file-input");
  const wordCount = document.getElementById("word-count");
  const summaryBox = document.getElementById("summary-output");
  const confidenceBox = document.getElementById("confidence-score");
  const semanticBox = document.getElementById("semantic-score");
  const loader = document.getElementById("loader");
  const clearBtn = document.getElementById("clear-btn");
  const downloadBtn = document.getElementById("download-btn");
  const themeSwitch = document.getElementById("theme-switch");
  const errorBox = document.getElementById("error-message");
  const summaryWordCount = document.getElementById("summary-word-count");
  const copyBtn = document.getElementById("copy-btn");
  
  // History elements
  const historySection = document.getElementById("history-section");
  const historyList = document.getElementById("history-list");
  const noHistory = document.getElementById("no-history");
  const clearHistoryBtn = document.getElementById("clear-history-btn");

  // Initialize history on page load
  loadHistory();

  // Word count
  textInput.addEventListener("input", () => {
    const words = textInput.value.trim().split(/\s+/).filter(Boolean).length;
    wordCount.textContent = `Words: ${words}`;
  });

  // Load theme from localStorage
  const savedTheme = localStorage.getItem("theme") || "dark";
  const savedColor = localStorage.getItem("theme-class") || "";
  if (savedTheme === "light") {
    document.body.classList.add("light-mode");
    themeSwitch.checked = true;
  }
  if (savedColor) {
    document.body.classList.add(savedColor);
  }

  // Theme switch toggle
  themeSwitch.addEventListener("change", () => {
    if (themeSwitch.checked) {
      document.body.classList.add("light-mode");
      localStorage.setItem("theme", "light");
    } else {
      document.body.classList.remove("light-mode");
      localStorage.setItem("theme", "dark");
    }
  });

  // Model selection highlight for Arabic
  const languageSelect = document.getElementById("language");
  const modelSelect = document.getElementById("model");
  function updateModelSelectUI() {
    if (languageSelect.value === "arabic") {
      modelSelect.classList.add("model-ignored");
      modelSelect.disabled = true;
    } else {
      modelSelect.classList.remove("model-ignored");
      modelSelect.disabled = false;
    }
  }
  languageSelect.addEventListener("change", updateModelSelectUI);
  updateModelSelectUI(); // Initial call

  // Form submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    summaryBox.textContent = "";
    confidenceBox.style.display = "none";
    semanticBox.style.display = "none";
    downloadBtn.style.display = "none";
    loader.style.display = "block";
    errorBox.style.display = "none";
    if (summaryWordCount) summaryWordCount.style.display = "none";
    if (copyBtn) copyBtn.style.display = "none";
    errorBox.textContent = "";

    const formData = new FormData(form);
    
    // Language selection handling - ensure it's properly set
    const selectedLanguage = languageSelect ? languageSelect.value : "english";
    formData.set("language", selectedLanguage);
    
    console.log("Form data:", {
      language: selectedLanguage,
      model: formData.get("model"),
      text: formData.get("text") ? "present" : "not present",
      file: formData.get("file") ? "present" : "not present"
    });

    // Large article warning
    let inputText = textInput.value.trim();
    if (!inputText && fileInput.files.length > 0) {
      // Try to read file size (approximate, since we can't read file content here)
      const file = fileInput.files[0];
      if (file && file.size > 500000) { // ~500KB, rough estimate for large text
        errorBox.style.display = "block";
        errorBox.classList.add("info-message");
        const errorText = errorBox.querySelector('.error-text');
        if (errorText) errorText.textContent = "ℹ️ Processing a large article. This may take longer than usual.";
      }
    } else if (inputText.split(/\s+/).length > 2000) {
      errorBox.style.display = "block";
      errorBox.classList.add("info-message");
      const errorText = errorBox.querySelector('.error-text');
      if (errorText) errorText.textContent = "ℹ️ Processing a large article. This may take longer than usual.";
    }

    if (!textInput.value.trim() && fileInput.files.length === 0) {
      loader.style.display = "none";
      errorBox.style.display = "block";
      errorBox.textContent = "Please enter text or upload a file.";
      summaryBox.textContent = "";
      return;
    }

    try {
      console.log("Sending request to /summarize...");
      const response = await fetch("/summarize", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Response received:", data);
      loader.style.display = "none";

      if (data.error) {
        if (errorBox) {
          errorBox.style.display = "block";
          const errorText = errorBox.querySelector('.error-text');
          if (errorText) errorText.textContent = data.error;
        }
        if (summaryBox) summaryBox.textContent = "";
        if (summaryWordCount) summaryWordCount.style.display = "none";
        if (copyBtn) copyBtn.style.display = "none";
      } else {
        if (errorBox) {
          errorBox.style.display = "none";
          const errorText = errorBox.querySelector('.error-text');
          if (errorText) errorText.textContent = "";
        }
        if (data.warning && errorBox) {
          errorBox.style.display = "block";
          const errorText = errorBox.querySelector('.error-text');
          if (errorText) errorText.textContent = "ℹ️ " + data.warning;
          errorBox.classList.add("info-message");
        } else if (errorBox) {
          errorBox.classList.remove("info-message");
        }
        if (summaryBox) {
          summaryBox.textContent = data.summary;
          summaryBox.classList.remove("fade-in");
          void summaryBox.offsetWidth;
          summaryBox.classList.add("fade-in");
        }
        // Word count
        if (summaryWordCount) {
          const wordCount = data.summary.trim().split(/\s+/).filter(Boolean).length;
          summaryWordCount.textContent = `Word Count: ${wordCount}`;
          summaryWordCount.style.display = "block";
        }
        if (copyBtn) {
          copyBtn.style.display = "inline-block";
          copyBtn.onclick = () => {
            navigator.clipboard.writeText(data.summary);
            copyBtn.textContent = "Copied!";
            setTimeout(() => { copyBtn.textContent = "Copy to Clipboard"; }, 1500);
          };
        }
        if (confidenceBox) {
          confidenceBox.style.display = "block";
          confidenceBox.textContent = `Confidence: ${data.confidence}`;
        }
        if (semanticBox) {
          semanticBox.style.display = "block";
          semanticBox.textContent = `Semantic Similarity: ${data.semantic_similarity}`;
        }
        if (downloadBtn) downloadBtn.style.display = "inline-block";
        
        // Save to history
        if (data.summary) {
          // Use backend's model_used if present, otherwise fallback to frontend logic
          let modelName = data.model_used || ((selectedLanguage === 'arabic') ? 'mT5' : formData.get("model"));
          saveToHistory({
            summary: data.summary,
            language: selectedLanguage,
            model: modelName,
            confidence: data.confidence,
            semantic_similarity: data.semantic_similarity,
            timestamp: new Date().toISOString()
          });
        }
      }
    } catch (err) {
      console.error("Error during summarization:", err);
      loader.style.display = "none";
      errorBox.style.display = "block";
      errorBox.textContent = "An error occurred: " + err.message;
      summaryBox.textContent = "";
    }
  });

  // Clear button
  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    fileInput.value = "";
    summaryBox.textContent = "Your summary will appear here...";
    confidenceBox.style.display = "none";
    semanticBox.style.display = "none";
    downloadBtn.style.display = "none";
    wordCount.textContent = "Words: 0";
    if (summaryWordCount) summaryWordCount.style.display = "none";
    if (copyBtn) copyBtn.style.display = "none";
  });

  // Dismiss error/info message
  document.querySelectorAll('.close-btn').forEach(btn => {
    btn.onclick = function() {
      if (this.parentElement) this.parentElement.style.display = 'none';
    };
  });

  // History Functions
  function saveToHistory(summaryData) {
    try {
      let history = JSON.parse(localStorage.getItem('summaryHistory') || '[]');
      
      // Add new summary to the beginning
      history.unshift(summaryData);
      
      // Keep only the last 20 summaries
      if (history.length > 20) {
        history = history.slice(0, 20);
      }
      
      localStorage.setItem('summaryHistory', JSON.stringify(history));
      loadHistory();
      console.log('Summary saved to history');
    } catch (error) {
      console.error('Error saving to history:', error);
    }
  }

  function loadHistory() {
    try {
      const history = JSON.parse(localStorage.getItem('summaryHistory') || '[]');
      
      if (history.length === 0) {
        if (noHistory) noHistory.style.display = 'block';
        if (historyList) historyList.style.display = 'none';
        return;
      }
      
      if (noHistory) noHistory.style.display = 'none';
      if (historyList) {
        historyList.style.display = 'block';
        historyList.innerHTML = '';
        
        history.forEach((item, index) => {
          const historyItem = createHistoryItem(item, index);
          historyList.appendChild(historyItem);
        });
      }
    } catch (error) {
      console.error('Error loading history:', error);
    }
  }

  function createHistoryItem(item, index) {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.setAttribute('data-index', index);
    
    const summary = document.createElement('div');
    summary.className = 'history-summary';
    summary.textContent = item.summary.length > 150 ? 
      item.summary.substring(0, 150) + '...' : item.summary;
    
    const meta = document.createElement('div');
    meta.className = 'history-meta';
    
    const time = document.createElement('span');
    time.className = 'history-time';
    time.textContent = formatTime(item.timestamp);
    
    const language = document.createElement('span');
    language.className = 'history-language';
    language.textContent = item.language.charAt(0).toUpperCase() + item.language.slice(1);
    
    const model = document.createElement('span');
    model.className = 'history-model';
    if (item.language && item.language.toLowerCase() === 'arabic') {
      model.textContent = 'Model: mT5';
    } else {
      model.textContent = `Model: ${item.model}`;
    }
    meta.appendChild(model);
    
    meta.appendChild(time);
    meta.appendChild(language);
    
    div.appendChild(summary);
    div.appendChild(meta);
    
    // Click to load summary
    div.addEventListener('click', () => {
      loadSummaryFromHistory(item);
      // Highlight selected item
      document.querySelectorAll('.history-item').forEach(item => {
        item.classList.remove('selected');
      });
      div.classList.add('selected');
    });
    
    return div;
  }

  function loadSummaryFromHistory(item) {
    if (summaryBox) {
      summaryBox.textContent = item.summary;
      summaryBox.classList.remove("fade-in");
      void summaryBox.offsetWidth;
      summaryBox.classList.add("fade-in");
    }
    
    if (summaryWordCount) {
      const wordCount = item.summary.trim().split(/\s+/).filter(Boolean).length;
      summaryWordCount.textContent = `Word Count: ${wordCount}`;
      summaryWordCount.style.display = "block";
    }
    
    if (confidenceBox) {
      confidenceBox.style.display = "block";
      confidenceBox.textContent = `Confidence: ${item.confidence}`;
    }
    
    if (semanticBox) {
      semanticBox.style.display = "block";
      semanticBox.textContent = `Semantic Similarity: ${item.semantic_similarity}`;
    }
    
    if (copyBtn) {
      copyBtn.style.display = "inline-block";
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(item.summary);
        copyBtn.textContent = "Copied!";
        setTimeout(() => { copyBtn.textContent = "Copy to Clipboard"; }, 1500);
      };
    }
    
    if (downloadBtn) downloadBtn.style.display = "inline-block";
  }

  function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    if (diffInHours < 1) {
      const diffInMinutes = Math.floor((now - date) / (1000 * 60));
      return `${diffInMinutes} minutes ago`;
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)} hours ago`;
    } else {
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
  }

  // Clear history button
  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear all summary history? This action cannot be undone.')) {
        localStorage.removeItem('summaryHistory');
        loadHistory();
        console.log('History cleared');
      }
    });
  }
});

// 🎨 Theme palette switcher
function switchTheme(themeClass) {
  document.body.classList.remove("theme-green", "theme-orange", "theme-blue");
  if (themeClass) {
    document.body.classList.add(themeClass);
    localStorage.setItem("theme-class", themeClass);
  } else {
    localStorage.removeItem("theme-class");
  }
}
