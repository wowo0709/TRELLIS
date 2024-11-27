var carousel_objects = {};


function make_carousel(carousel_id, item_template, items, num_to_display) {
    carousel_objects[carousel_id] = {};
    carousel_objects[carousel_id].item_template = item_template;
    carousel_objects[carousel_id].items = items;
    carousel_objects[carousel_id].num_to_display = num_to_display;
    carousel_objects[carousel_id].num_pages = Math.ceil(items.length / num_to_display);
    carousel_objects[carousel_id].current_page = 0;
    carousel_init(carousel_id);
}


function carousel_init(carousel_id) {
    let carousel = document.getElementById(carousel_id);
    let html = "";
    html += '<div class="x-carousel-slider">'
    for (let i = 0; i < carousel_objects[carousel_id].num_to_display; i++) {
        html += '<div class="x-carousel-slider-item">';
        if (i < carousel_objects[carousel_id].items.length)
            html += carousel_objects[carousel_id].item_template(carousel_objects[carousel_id].items[i]);
        html += '</div>';
    }
    html += '</div>';
    html += '<div class="x-carousel-nav">';
    html += `<div class="x-carousel-switch" onclick="carousel_prev('${carousel_id}')">\u25C0</div>`;
    html += '<div class="x-carousel-pages">';
    for (let i = 0; i < carousel_objects[carousel_id].num_pages; i++) {
        html += `<div class="x-carousel-page${i === 0 ? ' x-carousel-page-active' : ''}" onclick="carousel_page('${carousel_id}', ${i})"></div>`;
    }
    html += '</div>';
    html += `<div class="x-carousel-switch" onclick="carousel_next('${carousel_id}')">\u25B6</div>`;
    html += '</div>';
    carousel.innerHTML = html;
}


function carousel_render(carousel_id) {
    let carousel = document.getElementById(carousel_id);
    let slider = carousel.querySelector('.x-carousel-slider');
    num_to_display = carousel_objects[carousel_id].num_to_display;
    start_idx = carousel_objects[carousel_id].current_page * num_to_display;
    for (let i = 0; i < num_to_display; i++) {
        let item_idx = start_idx + i;
        let item = slider.children[i];
        if (item_idx < carousel_objects[carousel_id].items.length) {
            item.innerHTML = carousel_objects[carousel_id].item_template(carousel_objects[carousel_id].items[item_idx]);
        } else {
            item.innerHTML = "";
        }
    }
    carousel.querySelector('.x-carousel-slider').innerHTML = html;
}


function carousel_page(carousel_id, page) {
    let carousel = document.getElementById(carousel_id);
    carousel_objects[carousel_id].current_page = page;
    carousel.querySelector('.x-carousel-page-active').classList.remove('x-carousel-page-active');
    carousel.querySelector('.x-carousel-pages').children[page].classList.add('x-carousel-page-active');
    carousel_render(carousel_id);
}


function carousel_prev(carousel_id) {
    page = carousel_objects[carousel_id].current_page - 1;
    if (page < 0) page += carousel_objects[carousel_id].num_pages;
    carousel_page(carousel_id, page);
}

function carousel_next(carousel_id) {
    page = carousel_objects[carousel_id].current_page + 1;
    if (page >= carousel_objects[carousel_id].num_pages) page -= carousel_objects[carousel_id].num_pages;
    carousel_page(carousel_id, page);
}
