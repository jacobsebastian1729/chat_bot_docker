class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            inputField: document.querySelector('.chatbox__support input'),
            // uploadForm: document.getElementById('upload-form'),  // ❌ Disabled file upload form
            // fileInput: document.getElementById('file-upload')     // ❌ Disabled file input
        };

        this.state = false;
        this.messages = [];
        this.greetingSent = false;
    }

    display() {
        const { openButton, chatBox, sendButton, inputField } = this.args;

        // Ensure event listeners bind only once
        openButton.removeEventListener('click', () => this.toggleState(chatBox));
        sendButton.removeEventListener('click', () => this.onSendButton(chatBox));
        inputField.removeEventListener('keyup', this.handleKeyPress);
        // uploadForm?.removeEventListener('submit', (e) => this.handleFileUpload(e)); // ❌ Disabled upload form event

        // Set event listeners
        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        inputField.addEventListener('keyup', (e) => this.handleKeyPress(e, chatBox));
        // uploadForm?.addEventListener('submit', (e) => this.handleFileUpload(e));  // ❌ Disabled upload form event
    }
    
    toggleState(chatbox) {
        this.state = !this.state;

        if (this.state) {
            chatbox.classList.add('chatbox--active');
            this.args.chatBox.style.display = "flex";
            this.args.inputField.focus();

            if (!this.greetingSent) {
                this.sendGreeting(chatbox);
                this.greetingSent = true;

                document.body.classList.add('arc-reactor-border');
                setTimeout(() => {
                    document.body.classList.remove('arc-reactor-border');
                    document.documentElement.classList.add('arc-reactor-theme');
                }, 1000);

                const inputField = this.args.inputField;
                inputField.addEventListener('input', () => {
                    if (inputField.value.trim() !== "") {
                        document.body.classList.remove('arc-reactor-border');
                    }
                }, { once: true });
            }
        } else {
            chatbox.classList.remove('chatbox--active');
            setTimeout(() => {
                this.args.chatBox.style.display = "none";
            }, 400);
        }
    }

    sendGreeting(chatbox) {
        const introMessage = "Hi there! I'm Nova, your helpful assistant. How can I assist you today?";
        this.messages.push({ name: "Nova", message: introMessage });
        this.updateChatText(chatbox);
    }

    handleKeyPress(event, chatbox) {
        if (event.key === "Enter") {
            this.onSendButton(chatbox);
        }
    }

    onSendButton(chatbox) {
        const textField = this.args.inputField;
        const userMessage = textField.value.trim();

        if (userMessage === "") return;

        this.messages.push({ name: "User", message: userMessage });

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: userMessage }),
            mode: 'cors',
            headers: { 'Content-Type': 'application/json' },
        })
            .then(response => response.json())
            .then(data => {
                this.messages.push({ name: "Nova", message: data.answer });
                this.updateChatText(chatbox);
                textField.value = '';
            })
            .catch(error => {
                console.error('Error:', error);
                this.messages.push({ name: "Nova", message: "Oops! Something went wrong." });
                this.updateChatText(chatbox);
                textField.value = '';
            });
    }

    // ❌ Disabled file upload function
    /*
    handleFileUpload(event) {
        event.preventDefault();
        const fileInput = this.args.fileInput;

        if (fileInput.files.length === 0) {
            alert("Please select a file to upload.");
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.messages = []
                    this.messages.push({ name: "Nova", message: "File uploaded successfully! You can now ask questions based on the document." });
                } else {
                    this.messages.push({ name: "Nova", message: "File upload failed. Please try again." });
                }
                this.updateChatText(this.args.chatBox);
            })
            .catch(error => {
                console.error('Error:', error);
                this.messages.push({ name: "Nova", message: "Oops! File upload failed." });
                this.updateChatText(this.args.chatBox);
            });
    }
    */

    updateChatText(chatbox) {
        const chatMessageContainer = chatbox.querySelector('.chatbox__messages');
        chatMessageContainer.innerHTML = this.messages
            .slice().reverse().map(item =>
                `<div class="messages__item messages__item--${item.name === "Nova" ? 'visitor' : 'operator'}">
                    ${item.message}
                </div>`
            ).join('');

        chatMessageContainer.scrollTop = chatMessageContainer.scrollHeight;
    }
}

// Initialize the chatbox
const chatbox = new Chatbox();
chatbox.display();
