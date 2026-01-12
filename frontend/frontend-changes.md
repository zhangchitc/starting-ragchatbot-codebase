# Frontend Theme Toggle Feature

## Overview
Added a dark/light theme toggle feature to the Course Materials Assistant interface. Users can now switch between themes using a button in the top-right corner of the screen.

## Changes Made

### 1. HTML (`frontend/index.html`)
- Added `data-theme="dark"` attribute to the `<body>` element for initial theme state
- Added a new theme toggle button with sun and moon SVG icons positioned in the top-right corner

### 2. CSS (`frontend/style.css`)
- **New CSS Variables:**
  - `--code-bg`: For code block backgrounds (dark: `rgba(0, 0, 0, 0.2)`, light: `rgba(0, 0, 0, 0.06)`)
  - `--theme-toggle-bg`: Theme toggle button background
  - `--theme-toggle-hover`: Theme toggle button hover state

- **Light Theme Variables (`[data-theme="light"]`):**
  - `--background`: `#f8fafc` (light gray)
  - `--surface`: `#ffffff` (white)
  - `--surface-hover`: `#f1f5f9`
  - `--text-primary`: `#1e293b` (dark slate)
  - `--text-secondary`: `#64748b` (medium gray)
  - `--border-color`: `#e2e8f0`
  - `--assistant-message`: `#f1f5f9`
  - `--shadow`: Lighter shadow for light mode
  - `--welcome-bg`: `#eff6ff`
  - `--welcome-border`: `#bfdbfe`

- **Theme Toggle Button Styles:**
  - Fixed positioning in top-right corner
  - 44px circular button with border
  - Smooth transitions for all states (hover, active, focus)
  - Icon rotation and scale animations on theme change
  - Keyboard accessible with visible focus ring

- **Smooth Transitions:**
  - Added transitions to body and all theme-aware elements
  - 0.3s ease transition for background-color, color, and border-color

### 3. JavaScript (`frontend/script.js`)
- **New Functions:**
  - `initTheme()`: Initializes theme on page load, checking localStorage for saved preference
  - `toggleTheme()`: Switches between light and dark themes, saves to localStorage

- **Theme Persistence:**
  - Theme preference is saved to localStorage
  - Preference persists across page refreshes

## Accessibility Features
- Semantic `aria-label` on toggle button
- `title` attribute for additional context
- Full keyboard navigation support (Tab to focus, Enter/Space to activate)
- Visible focus ring using existing `--focus-ring` variable
- Sufficient color contrast maintained in both themes

## Visual Design
- Moon icon displayed in dark mode (clicking switches to light)
- Sun icon displayed in light mode (clicking switches to dark)
- Smooth 0.3s transition animations
- Icon rotates 180deg with scale effect during theme switch
- Button scales slightly on hover (1.05) and active press (0.95)

## Browser Compatibility
- Uses CSS custom properties (supported in all modern browsers)
- localStorage for persistence (supported in all modern browsers)
- SVG icons (universally supported)
