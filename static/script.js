document.getElementById('start-btn').addEventListener('click', () => {
    fetch('/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result-text').innerText = data.message;
        });
});

document.getElementById('stop-btn').addEventListener('click', () => {
    fetch('/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('result-text').innerText = data.message;
        });
});