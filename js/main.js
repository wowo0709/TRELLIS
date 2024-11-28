window.onload = function() {
    initWindow();
    make_carousel('results-txt2', txt2_carousel_item_template, null, txt2_items, 2, 4);
    make_carousel('results-img2', img2_carousel_item_template, null, img2_items, 2, 4);
    make_carousel('results-variants', variants_carousel_item_template, null, variants_items, 2, 1);

    prepare_scene_items().then(() => {
        make_carousel('results-scene', scene_carousel_item_template, null, scene_items, 1, 1);
    });
};
