const API_CONFIG = {
    REST_URL: null,
    WS_URL: null,
    DEFAULT_URL: window.localStorage.getItem('NGROK_URL') || 'http://localhost:8001'
};

// WebSocket connection
let ws = null;

// Initialize API configuration
async function initializeApiConfig() {
    const savedConfig = localStorage.getItem('API_CONFIG');
    if (savedConfig) {
        const config = JSON.parse(savedConfig);
        API_CONFIG.REST_URL = config.REST_URL;
        API_CONFIG.WS_URL = config.WS_URL;
    }
    
    if (!API_CONFIG.REST_URL) {
        let apiUrl = prompt("Please enter the API URL (from ngrok):", "");
        if (apiUrl) {
            apiUrl = apiUrl.replace(/\/$/, "");
            const wsUrl = apiUrl.replace(/^http/, 'ws') + '/ws';
            
            API_CONFIG.REST_URL = apiUrl;
            API_CONFIG.WS_URL = wsUrl;
            localStorage.setItem('API_CONFIG', JSON.stringify(API_CONFIG));
        }
    }
    
    if (API_CONFIG.WS_URL) {
        initializeWebSocket();
    }
}

// Initialize WebSocket connection
function initializeWebSocket() {
    if (!API_CONFIG.WS_URL) return;

    ws = new WebSocket(API_CONFIG.WS_URL);
    
    ws.onopen = () => {
        updateConnectionStatus(true);
        addSystemMessage('Connected to PHR Generative AI');
    };
    
    ws.onclose = () => {
        updateConnectionStatus(false);
        setTimeout(initializeWebSocket, 5000); // Retry connection after 5 seconds
    };
    
    ws.onerror = () => {
        updateConnectionStatus(false);
        addSystemMessage('Error connecting to chat server');
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleResponse(data);
        } catch (error) {
            addSystemMessage('Error processing server response');
        }
    };
}

// Update connection status display
function updateConnectionStatus(connected) {
    const statusDiv = document.getElementById('connection-status');
    statusDiv.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
    statusDiv.textContent = connected ? 'Connected' : 'Disconnected';
}

// Add system message to chat
function addSystemMessage(message) {
    const chatMessages = document.getElementById('chat-messages');
    const systemDiv = document.createElement('div');
    systemDiv.className = 'message system-message';
    systemDiv.textContent = message;
    chatMessages.appendChild(systemDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Create message element with enhanced formatting
function createMessageElement(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    
    // Format content with line breaks and markdown-like formatting
    if (!isUser && content.includes('\n')) {
        messageDiv.innerHTML = formatAssistantMessage(content);
    } else {
        messageDiv.textContent = content;
    }
    
    return messageDiv;
}

// Enhanced message formatting for assistant responses
function formatAssistantMessage(content) {
    return content
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/🎯|📊|🔍|💡|⚡|🏗️|🔧|📈|📐|🤖|⚠️|✅|🚀/g, '<span class="emoji">$&</span>')
        .replace(/•\s/g, '<span class="bullet">•</span> ');
}

// Handle sending questions
async function sendQuestion() {
    const questionInput = document.getElementById('question-input');
    const chatMessages = document.getElementById('chat-messages');
    const question = questionInput.value.trim();
    
    if (!question) return;
    
    chatMessages.appendChild(createMessageElement(question, true));
    questionInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    const messageData = {
        question: question,
        timestamp: new Date().toISOString()
    };
    
    try {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(messageData));
        } else {
            const response = await fetch(`${API_CONFIG.REST_URL}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(messageData),
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            hideTypingIndicator();
            handleResponse(data);
        }
    } catch (error) {
        hideTypingIndicator();
        addSystemMessage('Sorry, there was an error processing your request.');
    }
}

// Show typing indicator
function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'message assistant-message typing-indicator';
    typingDiv.innerHTML = '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span> AI is analyzing...';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Enhanced response handler for DCA and Wellbore data
function handleResponse(data) {
    const chatMessages = document.getElementById('chat-messages');
    hideTypingIndicator();
    
    if (data.type === 'error') {
        addSystemMessage(`Error: ${data.message}`);
        return;
    }
    
    const assistantContainer = document.createElement('div');
    assistantContainer.className = 'message assistant-message';
    
    // Enhanced explanation formatting
    if (data.explanation) {
        const explanationDiv = document.createElement('div');
        explanationDiv.className = 'explanation-content';
        explanationDiv.innerHTML = formatAssistantMessage(data.explanation);
        assistantContainer.appendChild(explanationDiv);
    }
    
    // Enhanced visualization handling for both DCA and Wellbore
    if (data.visualization) {
        const visualizationWrapper = document.createElement('div');
        visualizationWrapper.className = 'visualization-wrapper';
        
        const img = document.createElement('img');
        img.className = 'visualization';
        img.src = `data:image/png;base64,${data.visualization}`;
        img.alt = 'Generated visualization';
        
        // Add click to enlarge functionality
        img.addEventListener('click', () => {
            openImageModal(img.src);
        });
        
        // Determine visualization type based on content
        const vizTypeLabel = document.createElement('div');
        vizTypeLabel.className = 'visualization-label';
        
        if (data.explanation && data.explanation.includes('Wellbore')) {
            vizTypeLabel.textContent = '🏗️ Wellbore Diagram';
        } else if (data.explanation && data.explanation.includes('DCA')) {
            vizTypeLabel.textContent = '📊 DCA Analysis Chart';
        } else if (data.explanation && data.explanation.includes('ML')) {
            vizTypeLabel.textContent = '🤖 ML Prediction Chart';
        } else {
            vizTypeLabel.textContent = '📈 Production Analysis';
        }
        
        visualizationWrapper.appendChild(vizTypeLabel);
        visualizationWrapper.appendChild(img);
        assistantContainer.appendChild(visualizationWrapper);

        // Visualization explanation if available
        if (data.visualization_explanation) {
            const vizExp = document.createElement('div');
            vizExp.className = 'visualization-explanation';
            vizExp.innerHTML = formatAssistantMessage(data.visualization_explanation);
            assistantContainer.appendChild(vizExp);
        }
    }
    
    // SQL code display (for DCA queries that might include database queries)
    if (data.sql) {
        const sqlDiv = document.createElement('div');
        sqlDiv.className = 'sql-code';
        sqlDiv.innerHTML = `<pre><code>${data.sql}</code></pre>`;
        assistantContainer.appendChild(sqlDiv);
    }
    
    // Enhanced data table display
    if (data.data && data.data.length > 0) {
        const tableWrapper = document.createElement('div');
        tableWrapper.className = 'data-table-wrapper';
        
        const tableTitle = document.createElement('div');
        tableTitle.className = 'table-title';
        tableTitle.textContent = '📊 Data Results';
        tableWrapper.appendChild(tableTitle);
        
        const table = document.createElement('table');
        table.className = 'data-table';
        
        // Add headers
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        Object.keys(data.data[0]).forEach(key => {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Add data rows with enhanced formatting
        const tbody = document.createElement('tbody');
        data.data.slice(0, 10).forEach(row => { // Limit to first 10 rows for display
            const tr = document.createElement('tr');
            Object.values(row).forEach(value => {
                const td = document.createElement('td');
                // Format numbers and dates
                if (typeof value === 'number') {
                    td.textContent = value.toLocaleString();
                } else if (value && typeof value === 'string' && value.match(/^\d{4}-\d{2}-\d{2}/)) {
                    td.textContent = new Date(value).toLocaleDateString();
                } else {
                    td.textContent = value !== null ? value : '';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        
        // Add row count info if data is truncated
        if (data.data.length > 10) {
            const moreRowsInfo = document.createElement('div');
            moreRowsInfo.className = 'table-info';
            moreRowsInfo.textContent = `Showing first 10 of ${data.data.length} rows`;
            tableWrapper.appendChild(moreRowsInfo);
        }
        
        tableWrapper.appendChild(table);
        assistantContainer.appendChild(tableWrapper);
    }

    // Enhanced app URL handling with specific messaging
    if (data.app_url) {
        const appLink = document.createElement('div');
        appLink.className = 'external-app-link';
        
        // Determine app type and provide specific messaging
        let appType = '';
        let appIcon = '';
        let appDescription = '';
        
        if (data.app_url.includes('dca')) {
            appType = 'DCA Analysis';
            appIcon = '📊';
            appDescription = 'interactive decline curve analysis, detailed predictions, and advanced modeling';
        } else if (data.app_url.includes('wellbore')) {
            appType = 'Wellbore Visualization';
            appIcon = '🏗️';
            appDescription = 'interactive wellbore diagrams, component details, and 3D visualization';
        } else {
            appType = 'Detailed Analysis';
            appIcon = '🔍';
            appDescription = 'comprehensive analysis and advanced features';
        }
        
        appLink.innerHTML = `
            <div class="app-link-header">
                <span class="app-icon">${appIcon}</span>
                <strong>Open ${appType} App</strong>
            </div>
            <p class="app-description">For ${appDescription}:</p>
            <a href="${data.app_url}" target="_blank" rel="noopener noreferrer" class="app-link-button">
                🚀 Launch ${appType} App
            </a>
        `;
        assistantContainer.appendChild(appLink);
    }
    
    chatMessages.appendChild(assistantContainer);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Image modal for enlarged visualization viewing
function openImageModal(imageSrc) {
    // Create modal overlay
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
        cursor: pointer;
    `;
    
    // Create enlarged image
    const img = document.createElement('img');
    img.src = imageSrc;
    img.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        object-fit: contain;
        border-radius: 8px;
    `;
    
    modal.appendChild(img);
    
    // Close modal on click
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    document.body.appendChild(modal);
}

// Clear chat history
function clearChatHistory() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = '';
    addSystemMessage('Chat history cleared. Ready for new questions about DCA analysis or wellbore diagrams.');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeApiConfig();
    
    // Add event listener for config button
    const configBtn = document.getElementById('config-btn');
    if (configBtn) {
        configBtn.addEventListener('click', () => {
            localStorage.removeItem('API_CONFIG');
            if (ws) ws.close();
            initializeApiConfig();
        });
    }
    
    // Add event listener for clear chat button
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', clearChatHistory);
    }
    
    // Add event listener for Enter key
    document.getElementById('question-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendQuestion();
        }
    });
    
    // Auto-focus on input field
    document.getElementById('question-input').focus();
});