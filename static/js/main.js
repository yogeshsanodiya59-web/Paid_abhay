/**
 * main.js — MediPredict Client-side JavaScript
 * Handles: navbar scroll effect, alert auto-dismiss, table search
 */

document.addEventListener('DOMContentLoaded', function () {

  // ── Navbar scroll shadow ──────────────────────────────────
  const nav = document.getElementById('mainNav');
  if (nav) {
    window.addEventListener('scroll', function () {
      if (window.scrollY > 20) {
        nav.style.boxShadow = '0 4px 20px rgba(26,111,196,.12)';
      } else {
        nav.style.boxShadow = '0 1px 3px rgba(0,0,0,.08)';
      }
    });
  }

  // ── Auto-dismiss flash alerts after 5 seconds ─────────────
  const alerts = document.querySelectorAll('.alert.alert-dismissible');
  alerts.forEach(function (alert) {
    setTimeout(function () {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) bsAlert.close();
    }, 5000);
  });

  // ── Animate stat cards on dashboard ──────────────────────
  const statValues = document.querySelectorAll('.stat-card-value');
  if (statValues.length > 0) {
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }
      });
    }, { threshold: 0.1 });

    statValues.forEach(function (el) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(10px)';
      el.style.transition = 'all .4s ease';
      observer.observe(el);
    });
  }

  // ── Mini bars on dashboard table ──────────────────────────
  const miniBars = document.querySelectorAll('.mini-bar');
  miniBars.forEach(function (bar) {
    // Width is set inline via style attribute in template
    // Just ensure transition fires after paint
    const width = bar.style.width;
    bar.style.width = '0';
    setTimeout(function () { bar.style.width = width; }, 200);
  });

  // ── Fade-in cards on scroll ───────────────────────────────
  const fadeEls = document.querySelectorAll('.step-card, .feature-card, .stat-card');
  if ('IntersectionObserver' in window) {
    const fadeObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
          fadeObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    fadeEls.forEach(function (el, i) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      el.style.transition = `opacity .5s ease ${i * 0.08}s, transform .5s ease ${i * 0.08}s`;
      fadeObserver.observe(el);
    });
  }

});
