let conversationId = null;

document.addEventListener('DOMContentLoaded', function() {
    fetchModels();
    setupChat();
    setupModelSelection();
});


document.addEventListener('DOMContentLoaded', scrollToBottom);

function setupModelSelection() {
    const modelSelect = document.getElementById('modelSelect');
    const maxTokensInput = document.getElementById('maxTokens');
    const tokenLimitInfo = document.getElementById('tokenLimitInfo');
    const maxTokensInfo = document.getElementById('maxTokensInfo');

    modelSelect.addEventListener('change', function() {
        const selectedModel = this.options[this.selectedIndex].text.toLowerCase();

        let maxLimit, color, maxOutputTokens;

        if (selectedModel.includes('cohere')) {
            maxLimit = 4000;
            color = "#007bff"; 
            maxOutputTokens = 3400;
        } else if (selectedModel.includes('llama') || selectedModel.includes('meta')) {
            maxLimit = 8000;
            color = "#28a745"; 
            maxOutputTokens = 7400; 
        } else {
            tokenLimitInfo.textContent = "";
            maxTokensInfo.textContent = "";
            return;
        }

        tokenLimitInfo.textContent = `${selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1)} model: Max ${maxLimit} tokens (input + output)`;
        tokenLimitInfo.style.color = color;

        maxTokensInput.max = maxOutputTokens;
        maxTokensInput.value = Math.min(maxTokensInput.value, maxOutputTokens);
        maxTokensInfo.textContent = `Max output tokens: ${maxOutputTokens}`;
        maxTokensInfo.style.color = color;
    });
}

function fetchModels() {
    fetch('/list-models')
        .then(response => response.json())
        .then(data => {
            const modelSelect = document.getElementById('modelSelect');
            modelSelect.innerHTML = ''; 

            data.forEach(model => {
                if (model.capabilities.includes('CHAT') && !model.capabilities.includes('FINE_TUNE')) {
                    const option = document.createElement('option');
                    option.value = model.id;
                    option.textContent = model.displayName;
                    option.dataset.capabilities = JSON.stringify(model.capabilities);
                    modelSelect.appendChild(option);
                }
            });
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            const modelSelect = document.getElementById('modelSelect');
            modelSelect.innerHTML = `<option value="">Error loading models: ${error.message}</option>`;
        });
}

function uploadFiles() {
    const fileInput = document.getElementById('fileUpload');
    const files = fileInput.files;

    if (files.length === 0) {
        alert('Please select file(s) to upload');
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files[]', files[i]);
    }

    const uploadStatus = document.getElementById('upload-status');
    uploadStatus.textContent = 'Uploading...';

    fetch('/upload-to-oci', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            uploadStatus.textContent = `Error: ${data.error}`;
        } else {
            uploadStatus.textContent = `${data.status} (${data.processed}/${data.total})`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        uploadStatus.textContent = 'An error occurred while uploading the file(s)';
    });
}


document.addEventListener('DOMContentLoaded', scrollToBottom);

function setupChat() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-message');

    chatForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const message = userInput.value.trim();
        if (message) {
            appendMessage('You', message, 'user-message');
            userInput.value = '';
            botResponse(message);
        }
    });
}

function appendMessage(sender, text, className) {
    const chatMessages = document.getElementById('chat-messages');

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);

    const infoDiv = document.createElement('div');
    infoDiv.classList.add('message-info');
    infoDiv.textContent = sender;

    const textDiv = document.createElement('div');
    textDiv.classList.add('message-text');
    textDiv.textContent = text;

    messageDiv.appendChild(infoDiv);
    messageDiv.appendChild(textDiv);
    chatMessages.appendChild(messageDiv);

    moveTypingIndicator(); 
    scrollToBottom();
}

function showTypingIndicator() {
    let chatMessages = document.getElementById('chat-messages');
    let typingIndicator = document.getElementById('typing-indicator');

    
    if (!typingIndicator) {
        typingIndicator = document.createElement('div');
        typingIndicator.id = "typing-indicator";
        typingIndicator.classList.add('typing-indicator');
        typingIndicator.innerHTML = `
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        `;
        chatMessages.appendChild(typingIndicator);
    }

  
    typingIndicator.style.bottom = '18px'; //  Fixed bottom-left when first appearing
    typingIndicator.style.left = '10px'; 

    typingIndicator.style.display = 'flex';
    moveTypingIndicator(); 
}

function hideTypingIndicator() {
    let typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
}

function moveTypingIndicator() {
    let chatMessages = document.getElementById('chat-messages');
    let typingIndicator = document.getElementById('typing-indicator');
    let lastMessage = chatMessages.lastElementChild;

    if (typingIndicator && lastMessage) {
        let lastMessageBottom = lastMessage.offsetTop + lastMessage.offsetHeight;
        typingIndicator.style.top = lastMessageBottom + 'px'; 
    }

    scrollToBottom(); 
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function botResponse(userMessage) {
    showTypingIndicator(); 

    const selectedModel = document.getElementById('modelSelect');
    const selectedModelId = selectedModel.value;
    const selectedModelDisplayName = selectedModel.options[selectedModel.selectedIndex].text;

    if (!selectedModelId) {
        hideTypingIndicator();
        appendMessage('AI', "Please select a model before sending a message.", 'ai-message');
        return;
    }

    const params = {
        query: userMessage,
        model_id: selectedModelId,
        model_display_name: selectedModelDisplayName,
        max_tokens: parseInt(document.getElementById('maxTokens').value, 10),
        temperature: parseFloat(document.getElementById('temperature').value),
        frequency_penalty: parseFloat(document.getElementById('frequencyPenalty').value),
        top_p: parseFloat(document.getElementById('topP').value),
        top_k: parseInt(document.getElementById('topK').value, 10),
        conversation_id: conversationId
    };

    fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        hideTypingIndicator();

        if (data.error) {
            appendMessage('AI', `Error: ${data.error}`, 'ai-message');
        } else {
            appendMessage('AI', data.response, 'ai-message');
            conversationId = data.conversation_id;
        }

        moveTypingIndicator(); 
    })
    .catch(error => {
        console.error('Error:', error);
        hideTypingIndicator();
        appendMessage('AI', "An error occurred while processing your request.", 'ai-message');
    });

    scrollToBottom(); 
}


document.addEventListener('DOMContentLoaded', scrollToBottom);
