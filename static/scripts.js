document.addEventListener('DOMContentLoaded', function() {
    // Dynamic form validation
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
    
    // Risk meter animation
    const riskMeters = document.querySelectorAll('.risk-meter');
    riskMeters.forEach(meter => {
        const riskLevel = parseFloat(meter.dataset.risk);
        const fill = meter.querySelector('.risk-fill');
        
        // Animate fill
        let width = 0;
        const interval = setInterval(() => {
            if (width >= riskLevel) {
                clearInterval(interval);
            } else {
                width++;
                fill.style.width = width + '%';
                
                // Change color based on risk level
                if (width < 30) {
                    fill.style.backgroundColor = '#28a745'; // Green
                } else if (width < 70) {
                    fill.style.backgroundColor = '#ffc107'; // Yellow
                } else {
                    fill.style.backgroundColor = '#dc3545'; // Red
                }
            }
        }, 20);
    });
});