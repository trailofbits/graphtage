$( document ).ready(function (){

var createList = function(selector){

    var ul = $('<ul class="current"></ul>');
    var selected = $(selector);

    if (selected.length === 0){
        return;
    }

    selected.clone().each(function (i,e){

        var p = $(e).children('.descclassname');
        var n = $(e).children('.descname');
        var l = $(e).children('.headerlink');

        var a = $('<a>');
        a.attr('href',l.attr('href')).attr('title', 'Link to this definition');

        a.append(p).append(n);

        var entry = $('<li class="toctree-l4">').append(a);
        ul.append(entry);
    });
    return ul;
}

if($('dl.class > dt').length || $('dl.function > dt').length || $('dl.data > dt').length) {
    /* collapse any open menus */
    var menu = $('.wy-menu ul:first');
    menu.find('.current').removeClass("current");

    var pagename = $("h1")[0].innerText;

    if(pagename === "graphtage package") {
        pagename = "graphtage module";
    }

    var header = $('<li class="toctree-l2 current"><a class="reference internal" href="#">' + pagename + '</a></li>')
    var ul = $('<ul class="current"></ul>');
    header.append(ul);

    menu.find('ul:first').prepend(header);

    var x = [];
    x.push(['Classes','dl.class > dt']);
    x.push(['Functions','dl.function > dt']);
    x.push(['Variables','dl.data > dt']);

    var first = true;

    x.forEach(function (e) {
        var l = createList(e[1]);
        if (l) {
            var li = $('<li class="toctree-l3"><a class="reference internal" href="#">' + e[0] + '</a></li>')
            if(first) {
                li.addClass("current");
                first = false;
            }
            li.append(l);
            ul.append(li);
        }
    });
}

});
