// Create version selector for documentation top bar.
(function() {

  var url = window.location.href;
  var theano_dir = 'theano'; // directory containing theano doc
  // Default theano version: release and development.
  var versions_dir = {"release": "theano", "dev": "theano_versions/dev"};

  // If doc is run localy
  if (url.startsWith('file')) {
    theano_dir = 'html';
    versions_dir = {"local":"html", "test":"test"};
  }

  var root_url = url.substring(0, url.search('/' + theano_dir)) + '/';

  // Regular expression to find theano version directory in URL.
  var version_regex = new RegExp("\\/" + theano_dir + "(_versions\\/)?([_a-zA-Z.0-9]*)\\/");

  // Get current version
  var current_version = url.match(version_regex)[0]
  current_version = current_version.substring(1, current_version.length - 1)

  // Add current version in case versions.json is unavailable
  if (current_version != "theano" && current_version != "html") {
    ver = current_version.replace("theano_versions/", "")
    versions_dir[ver] = current_version
  }

  function build_vswitch() {
  // Build HTML string for version selector, based on ReadTheDocs theme's versions.html

    var vlabel = current_version.replace("theano_versions/", "");
    if (vlabel == 'theano') {
      vlabel = 'release';
    }
    var vswitch = ['<div class="rst-versions" data-toggle="rst-versions" role="note" aria-label="versions" align=left>'];
    vswitch.push('<span class="rst-current-version" data-toggle="rst-current-version">');
    vswitch.push('<span class="fa fa-book"></span>');
    vswitch.push('v: ', vlabel, ' ');
    vswitch.push('<span class="fa fa-caret-down"></span>');
    vswitch.push('</span>');

    vswitch.push('<div class="rst-other-versions">');

    vswitch.push('<dl>');
    vswitch.push('<dt>Versions</dt>');
    for (var version in versions_dir) {
      var new_url = url.replace(url.match(version_regex)[0], '/' + versions_dir[version] + '/');
      vswitch.push('<dd><a href=\"', new_url, '\">', version, '</a></dd>');
    }
    vswitch.push('</dl>');

    vswitch.push('<dl>');
    vswitch.push('<dt>Downloads</dt>');
    var pdf_url = root_url + current_version + "/theano.pdf"
    vswitch.push('<dd><a href=\"', pdf_url, '\">', 'PDF', '</a></dd>');
    vswitch.push('</dl>');

    vswitch.push('<dl>');
    vswitch.push('<dt>On GitHub</dt>');
    var git_master = "https://github.com/Theano/Theano"
    vswitch.push('<dd><a href=\"', git_master + '\">', 'Fork me', '</a></dd>');
    vswitch.push('</dl>');

    vswitch.push('</div>');
    vswitch.push('</div>');
    return vswitch.join('');
  }

  function build_vswitch_up() {
  // Build HTML string for version selector, based on ReadTheDocs theme's versions.html

    var vlabel = current_version.replace("theano_versions/", "");
    if (vlabel == 'theano') {
      vlabel = 'release';
    }
    else if (vlabel != "dev") {
        vlabel = '';
    }
    var vswitch = ['<div class="rst-versions-up" data-toggle="rst-versions" role="note" aria-label="versions" align=center>'];
    vswitch.push('<span class="rst-current-version" data-toggle="rst-current-version">');
    vswitch.push(vlabel);
    vswitch.push(' <span class="fa fa-caret-down"></span>');
    vswitch.push('</span>');
    vswitch.push('</div>');
    return vswitch.join('');
  }

// Create HTML for version switcher and assign to placeholder in layout.html.
  $(document).ready(function() {
    // Build default switcher
    $('.version_switcher_placeholder').html(build_vswitch());
    $('.version_switcher_placeholder_up').html(build_vswitch_up());

    // Check server for other doc versions and update switcher.
    if (url.startsWith('http')) {
      $.getJSON(root_url + 'theano_versions/versions.json', function(data){
        $.each(data, function(version, dir) {
            versions_dir[version] = dir;
        });
        $('.version_switcher_placeholder').html(build_vswitch());
        $('.version_switcher_placeholder_up').html(build_vswitch_up());
      });
    }
  });
})();
