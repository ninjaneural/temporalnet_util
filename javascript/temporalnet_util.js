
let check_complete_timer = null;

function temporalnet_util_cancel_video() {
    const start_btn = document.querySelector('#temporalnet_util_start_button');
    const cancel_btn = document.querySelector('#temporalnet_util_cancel_button');
    start_btn.disabled = true;
    cancel_btn.disabled = true;
    setTimeout(()=>{
        console.log('forced')
        if (start_btn.disabled) {
            start_btn.disabled = false;
        }
    }, 10000);
}

function temporalnet_util_process_video() {
    const start_btn = document.querySelector('#temporalnet_util_start_button');
    const cancel_btn = document.querySelector('#temporalnet_util_cancel_button');
    const status = document.querySelector('#temporalnet_util_status textarea');
    const check_complete = document.querySelector('#temporalnet_util_check_complete textarea');

    start_btn.disabled = true;
    cancel_btn.disabled = true;

    setTimeout(()=>{
        if (check_complete_timer) {
            clearInterval(check_complete_timer);
            check_complete_timer = null;
        }
        check_complete_timer = setInterval(function() {
            console.log('check_complete', check_complete.value);
            if (check_complete.value=='complete') {
                const start_btn = document.querySelector('#temporalnet_util_start_button');
                const cancel_btn = document.querySelector('#temporalnet_util_cancel_button');
                setTimeout(()=>{
                    start_btn.disabled = false;
                    cancel_btn.disabled = true;
                }, 1000);
                clearInterval(check_complete_timer);
                check_complete_timer = null;                
            }
        }, 1000);

        cancel_btn.disabled = false;
    }, 3000);

    return [...arguments];
}
