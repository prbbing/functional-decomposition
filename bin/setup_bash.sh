
BINDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPDIR=`dirname ${BINDIR}`

if [ ".$QUIET" == "." ]
then
    echo -e
    echo -e "\e[30m\e[107mIMPORTANT\e[0m: make sure you __source__ this script"
    echo -e "rather than execute it:"
    echo -e
    echo -e "\e[1m    yourprompt$ . \$FD_PATH/setup_bash.sh\e[0m"
    echo -e
    echo -e "otherwise the paths will not be correctly set.  If you'd like to"
    echo -e "silence this annoying message in the future, source the script"
    echo -e "like this:"
    echo -e
    echo -e "\e[1m    yourprompt$ QUIET=1 . \$FD_PATH/setup_bash.sh\e[0m"
    echo -e
    echo -e "Python package dependency check:"
    for PACKAGE in numpy scipy numexpr matplotlib ROOT
    do
        VERSION=$(python -c "import ${PACKAGE}; print ${PACKAGE}.__version__" 2>/dev/null )

        if [ $? -ne 0 ]
        then
            printf "%12s (%b)\n" "${PACKAGE}" "\e[31mnot found\e[0m"
        else
            printf "%12s (%b)\n" "${PACKAGE}" "\e[32mfound ${VERSION}\e[0m"
        fi
    done
    echo -e
    echo -e "ROOT is __optional__, and is required only to import Tree objects from"
    echo -e "  .root files.  If ROOT is not installed, you can still import .csv"
    echo -e "  files."
    echo
fi

[[ ":$PATH:"       != *${BINDIR}* ]] && export       PATH="${BINDIR}:${PATH}"
[[ ":$PYTHONPATH:" != *${TOPDIR}* ]] && export PYTHONPATH="${TOPDIR}:${PYTHONPATH}"

export FD_DIR=${TOPDIR}
