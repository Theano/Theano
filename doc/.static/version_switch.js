// Create version selector for documentation top bar.

(function() {

  var url = window.location.href;
  var theano_dir = 'theano'; // directory containing theano doc
  // Default theano version: release and development.
  var versions_dir = {"release": "theano", "dev": "theano_versions/dev"};

  // If doc is run localy
  if (url.startsWith('file')) {
    theano_dir = 'html';
    versions_dir = {"local":"html"};
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
  
  function build_select() {
  // Build HTML string for version selector combo box and
  // select current version by iterating versions_dir.

    var select = ['<select>'];
    $.each(versions_dir, function(version, dir){
      select.push('<option value="' + version + '"');
      if (dir == current_version)
        select.push(' selected="selected">' + version + '</option>');
      else
        select.push('>' + version + '</option>');
    });

    return select.join('');
  }

  function on_switch() {
  // Method triggered when an option is selected in combo box.
    var selected = $(this).children('option:selected').attr('value');

    // Insert selected version in URL.
    var new_url = url.replace(url.match(version_regex)[0],
                    '/' + versions_dir[selected] + '/');
    if (url != new_url) {
      $.ajax({
        success: function() {
          window.location.href = new_url;
        },
        // If page not in version, go to root of documentation.
        error: function() {
          window.location.href = root_url + versions_dir[selected] + '/';
        }
      });
    }
  }

// Create combobox HTML, assign to placeholder in layout.html and
// bind selection method.
  $(document).ready(function() {
    // Get theano version.
    // var current_version = DOCUMENTATION_OPTIONS.VERSION;

    // Build default switcher
    $('.version_switcher_placeholder').html(build_select());
    $('.version_switcher_placeholder select').bind('change', on_switch)

    // Check server for other doc versions and update switcher.
    if (url.startsWith('http')) {
      $.getJSON(root_url + 'theano_versions/versions.json', function(data){
        $.each(data, function(version, dir) {
            versions_dir[version] = dir;
        });

        $('.version_switcher_placeholder').html(build_select()); 
        $('.version_switcher_placeholder select').bind('change', on_switch)
      });
    }    
  });
})();
