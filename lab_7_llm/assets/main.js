document.getElementById('submit-btn').addEventListener('click', async () => {
    const text = document.getElementById('input-text').value.trim();
    if (!text) {
        document.getElementById('output').textContent = 'Please enter some text';
        return;
    }
    document.getElementById('output').textContent = 'Processing...';
    try {
        const response = await fetch('/infer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: text})
        });
        const data = await response.json();
        document.getElementById('output').textContent = data.infer || 'No result';
    } catch (error) {
        document.getElementById('output').textContent = 'Error: ' + error;
    }
});