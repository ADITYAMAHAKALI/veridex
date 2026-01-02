/* Custom JavaScript for Veridex documentation */

document.addEventListener('DOMContentLoaded', function() {
  // Add copy button feedback
  const copyButtons = document.querySelectorAll('.md-clipboard');
  copyButtons.forEach(button => {
    button.addEventListener('click', function() {
      const icon = this.querySelector('.md-clipboard__icon');
      if (icon) {
        icon.textContent = '✓';
        setTimeout(() => {
          icon.textContent = '';
        }, 2000);
      }
    });
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add external link indicator
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    if (!link.hostname.includes('adityamahakali.github.io')) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
    }
  });
});
