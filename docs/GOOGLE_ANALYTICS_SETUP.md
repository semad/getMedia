# 📊 Google Analytics Setup Guide

This guide will help you set up Google Analytics 4 (GA4) on your Telegram Media Analysis pages.

## 🚀 Quick Setup

### 1. Get Your Google Analytics Measurement ID

1. Go to [Google Analytics](https://analytics.google.com/)
2. Sign in with your Google account
3. Create a new property or use an existing one
4. Copy your **Measurement ID** (format: `G-XXXXXXXXXX`)

### 2. Update Your HTML Files

Run the provided script to automatically update all HTML files:

```bash
python update_ga_id.py G-XXXXXXXXXX
```

**Example:**
```bash
python update_ga_id.py G-ABC123DEF4
```

### 3. Test Your Setup

1. Open your HTML pages in a browser
2. Check the browser console for any errors
3. Use Google Analytics Real-Time reports to verify tracking

## 📁 Files Updated

The script will automatically update these files:
- `index.html` - Main analysis landing page
- `projects.html` - Project overview page  
- `telegram_analysis_dashboard.html` - Interactive dashboard

## 🔧 Manual Setup (Alternative)

If you prefer to manually update the files, replace `GA_MEASUREMENT_ID` with your actual ID in each HTML file:

```html
<!-- Google Analytics 4 -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'G-XXXXXXXXXX');
</script>
```

## 📊 What Gets Tracked

- **Page Views**: Every time someone visits your pages
- **User Sessions**: How long people stay on your site
- **Traffic Sources**: Where your visitors come from
- **Device Information**: Desktop, mobile, tablet usage
- **Geographic Data**: Where your visitors are located
- **User Behavior**: How people navigate your pages

## 🎯 Advanced Tracking (Optional)

### Custom Events

You can add custom tracking for specific interactions:

```javascript
// Track button clicks
gtag('event', 'click', {
    'event_category': 'engagement',
    'event_label': 'dashboard_launch'
});

// Track file downloads
gtag('event', 'download', {
    'event_category': 'engagement',
    'event_label': 'duplicates_csv'
});
```

### Enhanced Ecommerce (if applicable)

```javascript
gtag('event', 'view_item', {
    'currency': 'USD',
    'value': 0.00,
    'items': [{
        'item_id': 'telegram_analysis',
        'item_name': 'Telegram Media Analysis Dashboard'
    }]
});
```

## 🚨 Troubleshooting

### Common Issues

1. **No data appearing**: Wait 24-48 hours for first data
2. **Script errors**: Check browser console for JavaScript errors
3. **Ad blockers**: Some users may have analytics blocked
4. **Cache issues**: Clear browser cache and reload pages

### Verification Steps

1. **Check Network Tab**: Look for requests to `googletagmanager.com`
2. **Real-Time Reports**: Use GA4 Real-Time to see immediate data
3. **Tag Assistant**: Install Google Tag Assistant for debugging

## 📈 Analytics Dashboard

Once set up, you'll be able to see:

- **Real-Time**: Live visitor activity
- **Reports**: Daily, weekly, monthly traffic patterns
- **Audience**: Demographics and user behavior
- **Acquisition**: Traffic sources and campaigns
- **Behavior**: Page performance and user flow
- **Conversions**: Goal completions (if configured)

## 🔒 Privacy Considerations

- **GDPR Compliance**: Consider adding cookie consent banners
- **Data Retention**: GA4 has configurable data retention settings
- **User Privacy**: Respect user privacy preferences and browser settings

## 📚 Additional Resources

- [Google Analytics Help Center](https://support.google.com/analytics/)
- [GA4 Setup Guide](https://support.google.com/analytics/answer/10089681)
- [Google Tag Manager](https://tagmanager.google.com/) - Advanced tracking management
- [Google Analytics Academy](https://analytics.google.com/analytics/academy/) - Free courses

## 🎉 Success!

Once you've completed the setup:

1. ✅ Your pages will track visitor analytics
2. ✅ Data will appear in your GA4 dashboard within 24-48 hours
3. ✅ You can monitor traffic, engagement, and user behavior
4. ✅ Make data-driven decisions about your content and presentation

---

**Need Help?** Check the Google Analytics Help Center or consult the troubleshooting section above.
