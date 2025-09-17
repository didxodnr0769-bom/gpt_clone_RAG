import { useState } from "react";

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "안녕하세요! 무엇을 도와드릴까요?",
      sender: "bot",
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState(null);

  const sendMessageToServer = async (message) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/v1/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return response;
    } catch (error) {
      console.error("서버 통신 오류:", error);
      throw error;
    }
  };

  const handleStreamResponse = async (response) => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = "";

    // 스트리밍 메시지 생성
    const streamingMessage = {
      id: Date.now() + 1,
      text: "",
      sender: "bot",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, streamingMessage]);
    setStreamingMessageId(streamingMessage.id);

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.trim() === "") continue;

          try {
            // JSON 객체 파싱 (서버 응답 형식에 맞게)
            const parsed = JSON.parse(line);

            // chunk 필드가 있는 경우 (스트리밍 중)
            if (parsed.chunk !== undefined) {
              if (parsed.chunk) {
                fullText += parsed.chunk;

                // 실시간으로 메시지 업데이트
                setMessages((prev) =>
                  prev.map((msg) =>
                    msg.id === streamingMessage.id
                      ? { ...msg, text: fullText }
                      : msg
                  )
                );
              }

              // is_final이 true이면 스트림 완료
              if (parsed.is_final) {
                setStreamingMessageId(null);
                return;
              }
            }
          } catch (parseError) {
            // JSON 파싱 실패 시 무시 (불완전한 JSON일 수 있음)
            console.log("JSON 파싱 실패:", line);
          }
        }
      }
    } catch (error) {
      console.error("스트림 처리 오류:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === streamingMessage.id
            ? {
                ...msg,
                text: fullText + "\n\n[스트림 처리 중 오류가 발생했습니다.]",
              }
            : msg
        )
      );
    } finally {
      setStreamingMessageId(null);
    }
  };

  const handleSendMessage = async () => {
    if (inputText.trim() === "") return;

    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const currentInput = inputText;
    setInputText("");
    setIsLoading(true);

    try {
      const response = await sendMessageToServer(currentInput);

      // 스트림 응답 처리
      await handleStreamResponse(response);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "죄송합니다. 서버와의 통신에 문제가 발생했습니다.",
        sender: "bot",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* 헤더 */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-800">채팅</h1>
      </div>

      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.sender === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-white text-gray-800 border border-gray-200"
              }`}
            >
              <p className="text-sm">{message.text}</p>
              <p
                className={`text-xs mt-1 ${
                  message.sender === "user" ? "text-blue-100" : "text-gray-500"
                }`}
              >
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {/* 로딩 인디케이터 - 스트리밍 중이 아닐 때만 표시 */}
        {isLoading && !streamingMessageId && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 border border-gray-200 max-w-xs lg:max-w-md px-4 py-2 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.1s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                </div>
                <span className="text-xs text-gray-500">응답 중...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 입력 영역 */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <div className="flex space-x-4">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="메시지를 입력하세요..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            onClick={handleSendMessage}
            disabled={inputText.trim() === "" || isLoading}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading
              ? streamingMessageId
                ? "응답 중..."
                : "전송 중..."
              : "전송"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
