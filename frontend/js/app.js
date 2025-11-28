const navButtons = document.querySelectorAll('.nav-btn');
const pages = document.querySelectorAll('.page');

navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-target');

        navButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        pages.forEach(sec => {
            if (sec.id === targetId) {
                sec.classList.add('active');
            } else {
                sec.classList.remove('active');
            }
        });
    });
});


window.addEventListener('load', startCamera);

