(function () {
    const container = document.getElementById("chatMessagesContainer");
    const inputEl = document.getElementById("chatInput");
    const sendButton = document.getElementById("sendBtn");
    const chatEndpoint = "/chat.service";

    if (!container || !inputEl || !sendButton) {
        return;
    }

    function getCurrentTime() {
        const now = new Date();
        const hours = now.getHours().toString().padStart(2, "0");
        const minutes = now.getMinutes().toString().padStart(2, "0");
        return `${hours}:${minutes}`;
    }

    function createMessageComponent(type, content, timeStr = null) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message-item message-${type === "user" ? "user" : "assistant"}`;

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "message-bubble";
        bubbleDiv.textContent = content;

        const timeSpan = document.createElement("div");
        timeSpan.className = "message-time";
        const displayTime = timeStr || getCurrentTime();
        const senderLabel = type === "user" ? "你" : "小斋";
        timeSpan.textContent = `${displayTime} · ${senderLabel}`;

        messageDiv.appendChild(bubbleDiv);
        messageDiv.appendChild(timeSpan);
        return messageDiv;
    }

    function appendMessage(type, content, customTime = null) {
        const msgElement = createMessageComponent(type, content, customTime);
        container.appendChild(msgElement);
        container.scrollTo({
            top: container.scrollHeight,
            behavior: "smooth"
        });
    }

    function getLocalReply(rawText) {
        if (rawText.includes("你好") || rawText.includes("嗨") || rawText.includes("您好")) {
            return "安好。愿你今日心境如云般舒展。";
        }

        if (rawText.includes("谢谢") || rawText.includes("感谢")) {
            return "不须言谢，静听即是缘分。";
        }

        if (rawText.includes("?") || rawText.includes("？") || rawText.includes("如何") || rawText.includes("怎么")) {
            return "稍加思量，答案或许就在清风林影间。不妨静静再品一刻？";
        }

        if (rawText.length > 25) {
            return "读罢长句，心有涟漪。慢品其中味，也是一种安然。";
        }

        const gentleReplies = [
            "听闻，如风过竹梢。愿与你共守这份静好。",
            "素笺留白，余韵恰好。继续聊聊吧~",
            "浮生半日闲，得君一言，清欢有味。",
            "字句虽简，意蕴悠长。我在这里听着。"
        ];
        const randomIndex = Math.floor(Math.random() * gentleReplies.length);
        return gentleReplies[randomIndex];
    }

    function requestAssistantReply(rawText) {
        return fetch(chatEndpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ query: rawText })
        })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error("chat endpoint unavailable");
                }
                return response.json();
            })
            .then(function (result) {
                return result.ai_messages || getLocalReply(rawText);
            });
    }

    function resetTextareaHeight() {
        inputEl.style.height = "auto";
        inputEl.style.height = inputEl.value === "" ? "42px" : `${Math.min(inputEl.scrollHeight, 110)}px`;
    }

    function handleSendMessage() {
        const rawText = inputEl.value.trim();
        if (rawText === "") {
            return;
        }

        appendMessage("user", rawText);
        inputEl.value = "";
        resetTextareaHeight();

        requestAssistantReply(rawText)
            .then(function (replyText) {
                appendMessage("assistant", replyText);
            })
            .catch(function () {
                setTimeout(function () {
                    appendMessage("assistant", getLocalReply(rawText));
                }, 400);
            });
    }

    sendButton.addEventListener("click", function (event) {
        event.preventDefault();
        handleSendMessage();
    });

    inputEl.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
        }
    });

    inputEl.addEventListener("input", resetTextareaHeight);

    window.addEventListener("resize", function () {
        if (inputEl.value === "") {
            inputEl.style.height = "42px";
        }
    });

    resetTextareaHeight();
})();
