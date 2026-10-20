// Copy to pokemon-showdown/config/custom-formats.ts before starting the server.
export const Formats: import('../sim/dex-formats').FormatList = [
	{
		name: '[Gen 9] OU No Tera',
		mod: 'gen9',
		searchShow: false,
		challengeShow: true,
		ruleset: ['[Gen 9] OU', 'Terastal Clause'],
	},
];
