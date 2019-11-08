# Instructions to setup a new Reveal.js presentation


1) Create a new folder in `talks/` on the gh-pages branch

```
$ cd NewTalk
$ ln -s ../assets .
$ git submodule add https://github.com/EiffL/reveal.js.git
$ cp reveal.js/package.json .
$ wget https://raw.githubusercontent.com/EiffL/talks/master/template/index.html
$ wget https://raw.githubusercontent.com/EiffL/talks/master/template/gruntfile.js 
```

2) Now we are ready to build

```
$ npm install
```

3) Finally start the server

```
$ npm start
```


When you add your files to git DO NOT ADD `node_modules` !!!!!!!!!!!
