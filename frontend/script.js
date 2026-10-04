const API_URL = "http://127.0.0.1:5000";

async function analyzeCustomer() {
    const loading = document.getElementById("loading");
    const result = document.getElementById("result");
    const error = document.getElementById("error");
    const aiInsight = document.getElementById("aiInsight");

    loading.classList.remove("hidden");
    result.classList.add("hidden");
    error.classList.add("hidden");

    aiInsight.textContent = "Generating Gemini AI insights...";

    const data = {
        BALANCE: Number(document.getElementById("BALANCE").value),
        PURCHASES: Number(document.getElementById("PURCHASES").value),
        CASH_ADVANCE: Number(
            document.getElementById("CASH_ADVANCE").value
        ),
        CREDIT_LIMIT: Number(
            document.getElementById("CREDIT_LIMIT").value
        ),
        PAYMENTS: Number(
            document.getElementById("PAYMENTS").value
        ),
        PURCHASES_FREQUENCY: Number(
            document.getElementById("PURCHASES_FREQUENCY").value
        ),
        CASH_ADVANCE_FREQUENCY: Number(
            document.getElementById("CASH_ADVANCE_FREQUENCY").value
        ),
        PURCHASES_TRX: Number(
            document.getElementById("PURCHASES_TRX").value
        )
    };

    try {
        // -----------------------------
        // K-Means segmentation
        // -----------------------------

        const segmentResponse = await fetch(
            `${API_URL}/api/segment`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }
        );

        const segmentData = await segmentResponse.json();

        if (!segmentResponse.ok) {
            throw new Error(
                segmentData.error ||
                "Customer segmentation failed"
            );
        }

        document.getElementById("cluster").textContent =
            segmentData.cluster;

        document.getElementById("segment").textContent =
            segmentData.segment;

        // -----------------------------
        // Gemini AI insights
        // -----------------------------

        const aiResponse = await fetch(
            `${API_URL}/api/ai-insight`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    cluster: segmentData.cluster,
                    segment: segmentData.segment,
                    features: segmentData.features
                })
            }
        );

        const aiData = await aiResponse.json();

        if (!aiResponse.ok) {
            throw new Error(
                aiData.error ||
                "Gemini AI analysis failed"
            );
        }

        // Format Gemini markdown for display
        aiInsight.innerHTML = formatAIResponse(aiData.insight);

        result.classList.remove("hidden");

    } catch (err) {
        console.error("Error:", err);

        error.textContent =
            "Unable to complete the analysis. " +
            "Make sure the Flask backend is running and Gemini is available.";

        error.classList.remove("hidden");

    } finally {
        loading.classList.add("hidden");
    }
}


// -----------------------------------------
// Format Gemini response
// -----------------------------------------

function formatAIResponse(text) {
    let formatted = text;

    // Escape HTML characters
    formatted = formatted
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Convert bold markdown
    formatted = formatted.replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>"
    );

    // Convert numbered headings
    formatted = formatted.replace(
        /(?:^|\n)(\d+\.\s*[^:\n]+)(?:\s*)/g,
        "<div class=\"ai-heading\">$1</div>"
    );

    // Convert new lines
    formatted = formatted.replace(/\n/g, "<br>");

    return formatted;
}


// -----------------------------------------
// Reset form
// -----------------------------------------

function resetForm() {

    document.getElementById("BALANCE").value = 1000;
    document.getElementById("PURCHASES").value = 700;
    document.getElementById("CASH_ADVANCE").value = 400;
    document.getElementById("CREDIT_LIMIT").value = 4500;
    document.getElementById("PAYMENTS").value = 1700;
    document.getElementById("PURCHASES_FREQUENCY").value = 0.48;
    document.getElementById("CASH_ADVANCE_FREQUENCY").value = 0.10;
    document.getElementById("PURCHASES_TRX").value = 12;

    document.getElementById("result")
        .classList.add("hidden");

    document.getElementById("error")
        .classList.add("hidden");

    document.getElementById("loading")
        .classList.add("hidden");

    document.getElementById("cluster")
        .textContent = "-";

    document.getElementById("segment")
        .textContent = "-";

    document.getElementById("aiInsight")
        .textContent = "Generating AI insights...";
}