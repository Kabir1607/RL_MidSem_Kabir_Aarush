// Print the actual stats Showdown's simulator gives each Pokémon in the given team files, as JSON:
// {"<path>": [{"species": id, "stats": [hp, atk, def, spa, spd, spe]}, ...], ...}
// Usage: node tests/showdown_stats.js Teams/lowvar.txt Teams/opp_a.txt ...
const fs = require('fs');
const path = require('path');
const {Battle, Teams} = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));

const out = {};
for (const file of process.argv.slice(2)) {
	const battle = new Battle({formatid: 'gen9customgame'});
	battle.setPlayer('p1', {team: Teams.pack(Teams.import(fs.readFileSync(file, 'utf8')))});
	out[file] = battle.p1.pokemon.map(p => ({
		species: p.species.id,
		stats: [p.maxhp, p.storedStats.atk, p.storedStats.def, p.storedStats.spa, p.storedStats.spd, p.storedStats.spe],
	}));
}
console.log(JSON.stringify(out));
