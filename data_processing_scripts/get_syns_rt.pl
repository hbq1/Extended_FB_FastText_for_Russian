#!/usr/bin/perl -w

use strict;
use utf8;
 
use open ":utf8";
no warnings "utf8";

binmode (STDIN, ":utf8");
binmode (STDOUT, ":utf8");
binmode (STDERR, ":utf8");

die "Not enought arguments!" unless $ARGV[2];
my $vocab_freq_path = $ARGV[0];
my $syns_path = $ARGV[1];
my $out_info_path = $ARGV[2];

open VOCAB_F, '<', $vocab_freq_path // die "Can't open $vocab_freq_path";
binmode VOCAB_F, 'utf8';
open SYNS_F, '<', $syns_path // die "Can't open $syns_path";
open OUTP_F, '>', $out_info_path // die "Can't open $out_info_path";

my %vocab = ();
while (my $line = <VOCAB_F>) {
	chomp $line;
	$line =~ s/^\s+|\s+$//g;
	my ($freq, $word) = split(/\s+/, $line);
	$vocab{$word} = $freq;
}
close VOCAB_F;


my %h_info = ();
while (my $line = <SYNS_F>){
    chomp $line;
	$line =~ s/^\s+|\s+$//g;
	my ($word, @syns) = split(/\s+/, $line);
	unless ($vocab{$word}) {
		next;
	}

	for my $w(@syns) {
		$h_info{$w}{sum} += $vocab{$word};
		$h_info{$w}{cnt} += 1;
	}
}
close SYNS_F;

for my $word(keys %h_info) {
	print OUTP_F "$word\t$h_info{$word}{cnt}\t$h_info{$word}{sum}\n";
}

close OUTP_F;

