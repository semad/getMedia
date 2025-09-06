
// Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    initializeAnalytics();
});

function initializeCharts() {
    // Get chart data from script tag
    const chartDataScript = document.getElementById('chart-data');
    if (!chartDataScript) {
        console.error('Chart data script not found');
        return;
    }
    
    let chartData;
    try {
        chartData = JSON.parse(chartDataScript.textContent);
    } catch (error) {
        console.error('Error parsing chart data:', error);
        return;
    }
    
    // Initialize Chart.js charts
    const chartElements = document.querySelectorAll('.chart-container');
    
    chartElements.forEach((element, index) => {
        const chartType = element.dataset.chartType;
        console.log('Processing chart element:', chartType, 'element:', element);
        
        if (chartType && chartData[chartType]) {
            try {
                const chartConfig = chartData[chartType];
                console.log('Chart config for', chartType, ':', chartConfig);
                new Chart(element, chartConfig);
                console.log('Chart initialized successfully:', chartType);
            } catch (error) {
                console.error('Error creating chart for', chartType, ':', error);
            }
        } else {
            console.warn('No chart data found for type:', chartType, 'Available types:', Object.keys(chartData));
        }
    });
}

function initializeAnalytics() {
    // Google Analytics initialization
    
    if (typeof gtag !== 'undefined') {
        gtag('event', 'dashboard_loaded', {
            'event_category': 'dashboard',
            'event_label': 'index_page'
        });
    }
    
}


function handleChartInteraction(chartType, action, channelName) {
    
    if (typeof gtag !== 'undefined') {
        gtag('event', 'chart_interaction', {
            'event_category': 'dashboard',
            'event_label': `${chartType}_${action}_${channelName}`
        });
    }
    
}
