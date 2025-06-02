const API_CONFIG = {
    REST_URL: null,
    WS_URL: null,
    DEFAULT_URL: window.localStorage.getItem('SERVER_URL') || 'http://localhost:8001'
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
        let apiUrl = prompt("Please enter the Server URL (e.g., http://localhost:8001 or ngrok URL):", API_CONFIG.DEFAULT_URL);
        if (apiUrl) {
            apiUrl = apiUrl.replace(/\/$/, "");
            const wsUrl = apiUrl.replace(/^http/, 'ws') + '/ws';
            
            API_CONFIG.REST_URL = apiUrl;
            API_CONFIG.WS_URL = wsUrl;
            localStorage.setItem('API_CONFIG', JSON.stringify(API_CONFIG));
            localStorage.setItem('SERVER_URL', apiUrl);
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
        addSystemMessage('Connected to PHR Generative AI server');
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

// Create message element
function createMessageElement(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    messageDiv.innerHTML = content.replace(/\n/g, '<br>');
    return messageDiv;
}

// Handle sending questions
async function sendQuestion() {
    const questionInput = document.getElementById('question-input');
    const chatMessages = document.getElementById('chat-messages');
    const question = questionInput.value.trim();
    
    if (!question) return;
    
    // Show user message
    chatMessages.appendChild(createMessageElement(question, true));
    questionInput.value = '';
    
    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant-message typing-indicator';
    typingDiv.innerHTML = 'AI sedang mengetik<span class="dots">...</span>';
    typingDiv.id = 'typing-indicator';
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    const messageData = {
        question: question,
        timestamp: new Date().toISOString()
    };
    
    try {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(messageData));
        } else {
            // Fallback to REST API
            const response = await fetch(`${API_CONFIG.REST_URL}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(messageData),
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            
            // Remove typing indicator
            const typingIndicator = document.getElementById('typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            handleResponse(data);
        }
    } catch (error) {
        // Remove typing indicator on error
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        addSystemMessage('Maaf, terjadi kesalahan saat memproses permintaan Anda.');
        console.error('Error:', error);
    }
}

// Handle response from server
function handleResponse(data) {
    const chatMessages = document.getElementById('chat-messages');
    
    // Remove typing indicator if exists
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
    
    if (data.type === 'error') {
        addSystemMessage(`Error: ${data.message}`);
        return;
    }
    
    const assistantContainer = document.createElement('div');
    assistantContainer.className = 'message assistant-message';
    
    // Handle main explanation/response
    if (data.explanation) {
        const explanationDiv = document.createElement('div');
        explanationDiv.innerHTML = data.explanation.replace(/\n/g, '<br>');
        assistantContainer.appendChild(explanationDiv);
    }
    
    // Handle SQL code display (if applicable)
    if (data.sql) {
        const sqlDiv = document.createElement('div');
        sqlDiv.className = 'sql-code';
        sqlDiv.textContent = data.sql;
        assistantContainer.appendChild(sqlDiv);
    }
    
    // Handle visualization (if applicable)
    if (data.visualization) {
        const img = document.createElement('img');
        img.className = 'visualization';
        img.src = `data:image/png;base64,${data.visualization}`;
        assistantContainer.appendChild(img);

        if (data.visualization_explanation) {
            const vizExp = document.createElement('div');
            vizExp.className = 'visualization-explanation';
            vizExp.textContent = data.visualization_explanation;
            assistantContainer.appendChild(vizExp);
        }
    }
    
    // Handle tabular data (if applicable)
    if (data.data && data.data.length > 0) {
        const tableWrapper = document.createElement('div');
        tableWrapper.className = 'data-table-wrapper';
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
        
        // Add data rows
        const tbody = document.createElement('tbody');
        data.data.forEach(row => {
            const tr = document.createElement('tr');
            Object.values(row).forEach(value => {
                const td = document.createElement('td');
                td.textContent = value !== null ? value : '';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        
        tableWrapper.appendChild(table);
        assistantContainer.appendChild(tableWrapper);
    }

    // Handle external app links (if applicable)
    if (data.app_url) {
        const appLink = document.createElement('div');
        appLink.className = 'external-app-link';
        
        let appType = 'detailed analysis';
        if (data.app_url.includes('dca')) {
            appType = 'DCA analysis';
        } else if (data.app_url.includes('wellbore')) {
            appType = 'wellbore visualization';
        }
        
        appLink.innerHTML = `
            <p>For more detailed ${appType}, open the app: <a href="${data.app_url}" target="_blank" rel="noopener noreferrer">${data.app_url}</a></p>
        `;
        assistantContainer.appendChild(appLink);
    }
    
    chatMessages.appendChild(assistantContainer);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Clear chat history
function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = '';
    addSystemMessage('Chat history cleared');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeApiConfig();
    
    // Add event listener for config button
    const configBtn = document.getElementById('config-btn');
    if (configBtn) {
        configBtn.addEventListener('click', () => {
            localStorage.removeItem('API_CONFIG');
            localStorage.removeItem('SERVER_URL');
            if (ws) ws.close();
            initializeApiConfig();
        });
    }
    
    // Add event listener for Enter key
    document.getElementById('question-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendQuestion();
        }
    });
    
    // Add some styling for typing indicator
    const style = document.createElement('style');
    style.textContent = `
        .typing-indicator {
            opacity: 0.7;
        }
        .typing-indicator .dots {
            animation: blink 1.4s infinite;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
    `;
    document.head.appendChild(style);
});