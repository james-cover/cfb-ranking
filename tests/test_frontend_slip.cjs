const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../frontend/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Entire shipped script must parse.
const body = {innerHTML: ''};
const nodes = {slipBody: body, slipCount: {}, betSlip: {classList: {add(){}}}};
const state = {games: [{game_id: 7, home_team: 'A', away_team: 'B',
  home_spread: -3.5, home_spread_odds: -110, cover_probability_home: null}],
  slip: [], parlayWager: 10};
const context = vm.createContext({state, console,
  $: selector => nodes[selector.slice(1)], $$: () => [],
  saveSlip(){}, toast(){}, set(){}, games(){},
  esc: String, signed: String, odds: String, cash: String,
  num: Number,
});
const names = ['const decimalOdds=', 'const combinedAmerican=', 'const gameKey=',
  'function addPick(', 'function renderSlip('];
vm.runInContext(script.split('\n').filter(line => names.some(n => line.trim().startsWith(n))).join('\n'), context);
vm.runInContext("addPick('7', 'home', 'spread'); renderSlip();", context);
assert.equal(state.slip.length, 1);
assert.equal(state.slip[0].model_probability, null);
assert.match(body.innerHTML, /Model unavailable/);
assert.match(body.innerHTML, /Unavailable/);
assert.match(body.innerHTML, /\$19\.09/); // Payout still works without model confidence.
state.slip[0].model_probability = 60;
state.slip.push({...state.slip[0], id: '7:moneyline:home', market: 'moneyline'});
vm.runInContext('renderSlip();', context);
assert.match(body.innerHTML, /Unavailable/); // Do not multiply same-game probabilities.
console.log('Frontend slip checks passed');
