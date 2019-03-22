$style_config_version = '2.1.0'

desc "Install style config"
task :install_style_config do
  FileUtils.rm_rf "bodylabs-python-style" if Dir.exists? "bodylabs-python-style"
  raise unless system "git clone https://github.com/bodylabs/bodylabs-python-style.git"
  Dir.chdir 'bodylabs-python-style' do
    raise unless system "git checkout tags/#{$style_config_version}"
  end
end

task :require_style_config do
  Rake::Task[:install_style_config].invoke unless File.executable? 'bodylabs-python-style/bin/pylint_test'
end

$mac_os = `uname -s`.strip == 'Darwin'

desc "Install dependencies for distribution"
task :install_dist do
  if $mac_os
    raise unless system "brew update"
    raise unless system "brew install pandoc"
    raise unless system "pip install pypandoc"
  else
    puts
    puts "You must install:"
    puts
    puts " - pandoc"
    puts " - pypandoc"
    puts
    raise
  end
end

def command_is_in_path?(command)
  system("which #{ command} > /dev/null 2>&1")
end

task :unittest do
  if command_is_in_path? "nose2-2.7"
    raise unless system "nose2-2.7 --attribute '!missing_assets'"
  else
    raise unless system "nose2 --attribute '!missing_assets'"
  end
end

task :lint => :require_style_config do
  raise unless system "bodylabs-python-style/bin/pylint_test blmath --min_rating 10.0"
end

desc "Remove .pyc files"
task :clean do
  system "find . -name '*.pyc' -delete"
end

task :sdist do
  unless command_is_in_path? 'pandoc'
    puts
    puts "Please install pandoc."
    puts
    raise
  end
  raise unless system "python setup.py sdist"
end

task :upload do
  unless command_is_in_path?('pandoc')
    puts
    puts "Please install pandoc."
    puts
    raise
  end
  raise unless system "rm -rf dist"
  raise unless system "python setup.py sdist"
  raise unless system "twine upload dist/*"
end
