# Efficiency Hacks
[Mac](efficiency_hacks.md#mac)  
[Slack](efficiency_hacks.md#slack)  
[Sublime](efficiency_hacks.md#sublime)  
[Terminal](efficiency_hacks.md#terminal)  
[Github](efficiency_hacks.md#github)  

### Mac
- Set curser speed to maximum (helps when using arrow keys to navigate)
- Command + arrow key
  - Navigates to left/right/top/bottom of line/document
- Option + left/right arrow key
  - Navigates to next punctuation symbol

### Slack
- Command + k
  - Quick search for a channel
- Up arrow key (when in text prompt)
  - Edits your last message
- Option + click
  - Marks a message as unread so you can come back to it when you have more time
- Preferences
  - Mute all sounds
  - Notify me about "Nothing" (personal preference)
- Sidebar
  - Choose "Unreads and Starred conversations". This cuts down on unnecessary channels.
- Search
  - You can select channels you don't want to search (such as very noisy channels like alerts)

### Sublime
- Start sublime from your terminal
  - https://olivierlacan.com/posts/launch-sublime-text-3-from-the-command-line/
- Control + Shift + l
  - Places a cursor at every highlighted line
  - One use case is adding a comma to the end of every line
  - See below for another use case
- Control + d
  - Similar to "find all", but more powerful.
  - To use, highlight a text string. Then press control + d as many times as duplicate items you want to select.
  - A great use case is combining this with (Control + Shift + l). You can find all the items you want (say a misspelling) and edit every item at once.
- Command + Control + g
  - Same as above command, but selects all instances instantly
- Control + Shift + left/right arrow key
  - Highlights to next punctuation symbol or capital letter (useful to highlight part of a camelCase variable)
- Option + Shift + left/right arrow key
  - Highlights to next punctuation symbol (but also ignores under_case and camelCase)
- Command + Shift + left/right arrow key
  - Highlights entire line to left/right of cursor
  - Highlight text, then Command + k, then Command + l/u
  - Converts text to upper/lower case

### Terminal
- Control + a
  - move cursor to start of line
- Control + e
  - move cursor to end of line

### Github
- Autocomplete branches (put in .bash_profile)
```
# The following enables autocomplete for git commands/branches/etc.
if [ -f ~/.git-completion.bash ]; then
  . ~/.git-completion.bash
fi
```

- [Show which branch you are currently on in terminal (put in .bash_profile)](https://gist.github.com/joseluisq/1e96c54fa4e1e5647940)  
```
# Git branch in prompt.
parse_git_branch() {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

export PS1="\u@\h \W\[\033[32m\]\$(parse_git_branch)\[\033[00m\] $ "
```